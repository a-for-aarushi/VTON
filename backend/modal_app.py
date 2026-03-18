import modal
import io
import base64
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = modal.App("viton-hd-tryon")

viton_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "libgl1-mesa-glx", "libglib2.0-0", "libgomp1"])
    .pip_install([
        "torch==2.1.2",
        "torchvision==0.16.2",
        "Pillow",
        "numpy<2",
        "opencv-python-headless",
        "fastapi",
        "python-multipart",
        "pydantic",
        "scipy",
        "einops",
        "PyYAML",
        "torchgeometry",
        "onnxruntime",
        "rtmlib",
        "huggingface_hub==0.23.4",
        "transformers==4.41.2",
        "accelerate==0.30.1",
    ])
)

volume = modal.Volume.from_name("viton-hd-weights", create_if_missing=True)
CHECKPOINT_DIR = Path("/weights/weights")


# ── Utilities ─────────────────────────────────────────────────────
def b64_to_pil(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def resize_image(img: Image.Image, size=(768, 1024)) -> Image.Image:
    return img.resize(size, Image.LANCZOS)


def generate_cloth_mask(cloth_img: Image.Image) -> Image.Image:
    import cv2
    img_np = np.array(cloth_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return Image.fromarray(mask)


class Opt:
    def __init__(self):
        self.semantic_nc           = 13
        self.init_type             = "xavier"
        self.init_variance         = 0.02
        self.norm_G                = "spectralaliasinstance"
        self.ngf                   = 64
        self.num_upsampling_layers = "most"
        self.grid_size             = 5
        self.load_width            = 768
        self.load_height           = 1024


@app.cls(
    image=viton_image,
    gpu="T4",
    volumes={"/weights": volume},
    timeout=600,
    scaledown_window=300,
)
class VitonHDModel:

    @modal.enter()
    def load_model(self):
        import torch
        import sys
        import subprocess

        # Clone VITON-HD
        subprocess.run(["rm", "-rf", "/root/VITON-HD"], check=True)
        subprocess.run(
            ["git", "clone", "https://github.com/shadow2496/VITON-HD.git", "/root/VITON-HD"],
            check=True, capture_output=True, text=True
        )
        sys.path.insert(0, "/root/VITON-HD")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # ── Load VITON-HD models ───────────────────────────────────
        from networks import SegGenerator, GMM, ALIASGenerator
        opt = Opt()

        self.seg_model = SegGenerator(
            opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc
        ).to(self.device)

        self.gmm_model = GMM(opt, inputA_nc=7, inputB_nc=3).to(self.device)

        opt.semantic_nc = 7
        self.alias_model = ALIASGenerator(opt, input_nc=9).to(self.device)
        opt.semantic_nc = 13
        self.opt = opt

        from utils import load_checkpoint
        load_checkpoint(self.seg_model,   str(CHECKPOINT_DIR / "seg_final.pth"))
        load_checkpoint(self.gmm_model,   str(CHECKPOINT_DIR / "gmm_final.pth"))
        load_checkpoint(self.alias_model, str(CHECKPOINT_DIR / "alias_final.pth"))

        self.seg_model.eval()
        self.gmm_model.eval()
        self.alias_model.eval()
        print("✅ VITON-HD models loaded!")

        # ── Load pose estimator (RTMPose via rtmlib) ───────────────
        from rtmlib import Body
        self.pose_estimator = Body(
            det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
            det_input_size=(640, 640),
            pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
            pose_input_size=(192, 256),
            to_openpose=True,
            backend='onnxruntime',
            device='cpu',
        )
        print("✅ Pose estimator loaded!")

        # ── Load human parser (SegFormer trained on human parsing) ─
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        self.parse_processor = SegformerImageProcessor.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        )
        self.parse_model = SegformerForSemanticSegmentation.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        ).to(self.device)
        self.parse_model.eval()
        print("✅ Human parser loaded!")

    def get_pose(self, person_img: Image.Image, H=1024, W=768):
        """Run RTMPose and return pose RGB image."""
        import cv2
        import numpy as np

        img_np = np.array(person_img)

        # RTMPose returns keypoints in OpenPose format when to_openpose=True
        keypoints, scores = self.pose_estimator(img_np)

        # Draw skeleton on black canvas
        pose_rgb = Image.new("RGB", (W, H), (0, 0, 0))
        draw = ImageDraw.Draw(pose_rgb)

        # OpenPose connections (18 keypoints)
        # 0=nose,1=neck,2=RShoulder,3=RElbow,4=RWrist,
        # 5=LShoulder,6=LElbow,7=LWrist,8=RHip,9=RKnee,
        # 10=RAnkle,11=LHip,12=LKnee,13=LAnkle,14=REye,
        # 15=LEye,16=REar,17=LEar
        connections = [
            (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
            (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
            (0,14),(14,16),(0,15),(15,17)
        ]
        colors = [
            (255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),
            (85,255,0),(0,255,0),(0,255,85),(0,255,170),(0,255,255),
            (0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),
            (255,0,255),(255,0,170),(255,0,85)
        ]

        if keypoints is not None and len(keypoints) > 0:
            kps = keypoints[0]  # first person
            pts = {}
            for idx, (x, y) in enumerate(kps):
                if x > 0 and y > 0:
                    pts[idx] = (int(x), int(y))

            for i, (s, e) in enumerate(connections):
                if s in pts and e in pts:
                    color = colors[i % len(colors)]
                    draw.line([pts[s], pts[e]], fill=color, width=4)

            for idx, pt in pts.items():
                r = 5
                draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r],
                             fill=colors[idx % len(colors)])

        return pose_rgb

    def get_parse(self, person_img: Image.Image, H=1024, W=768):
        """
        Run SegFormer human parser.
        mattmdjaga/segformer_b2_clothes has these labels:
        0=Background, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes,
        5=Skirt, 6=Pants, 7=Dress, 8=Belt, 9=Left-shoe, 10=Right-shoe,
        11=Face, 12=Left-leg, 13=Right-leg, 14=Left-arm, 15=Right-arm,
        16=Bag, 17=Scarf

        Map to VITON-HD 13 classes:
        0=bg, 1=hat, 2=hair, 3=glove, 4=sunglasses, 5=upper,
        6=dress, 7=coats, 8=socks, 9=pants, 10=jumpsuits,
        11=scarf, 12=skirt
        """
        import torch
        import torch.nn.functional as F

        inputs = self.parse_processor(images=person_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.parse_model(**inputs)

        logits = outputs.logits  # (1, num_labels, H/4, W/4)
        upsampled = F.interpolate(
            logits, size=(H, W), mode='bilinear', align_corners=False
        )
        pred = upsampled.argmax(dim=1)[0].cpu().numpy()  # (H, W)

        # Map segformer labels → VITON-HD 13 classes
        # segformer: 0=bg,1=hat,2=hair,3=sunglasses,4=upper,5=skirt,
        #            6=pants,7=dress,8=belt,9=Lshoe,10=Rshoe,11=face,
        #            12=Lleg,13=Rleg,14=Larm,15=Rarm,16=bag,17=scarf
        #
        # viton-hd:  0=bg,1=hat,2=hair,3=glove,4=sunglasses,5=upper,
        #            6=dress,7=coat,8=socks,9=pants,10=jumpsuits,
        #            11=scarf,12=skirt
        seg_map = {
            0: 0,   # bg → bg
            1: 1,   # hat → hat
            2: 2,   # hair → hair
            3: 4,   # sunglasses → sunglasses
            4: 5,   # upper → upper-clothes ← KEY
            5: 12,  # skirt → skirt
            6: 9,   # pants → pants
            7: 6,   # dress → dress
            8: 0,   # belt → bg
            9: 0,   # left-shoe → bg
            10: 0,  # right-shoe → bg
            11: 0,  # face → bg (not in viton classes, use bg)
            12: 0,  # left-leg → bg
            13: 0,  # right-leg → bg
            14: 0,  # left-arm → bg (arms handled separately)
            15: 0,  # right-arm → bg
            16: 0,  # bag → bg
            17: 11, # scarf → scarf
        }

        # Build 13-class parse map
        parse_13 = np.zeros((H, W), dtype=np.int64)
        for src, dst in seg_map.items():
            parse_13[pred == src] = dst

        # Convert to 13-channel one-hot tensor
        parse_tensor = torch.zeros(1, 13, H, W, dtype=torch.float32)
        for c in range(13):
            parse_tensor[0, c] = torch.from_numpy((parse_13 == c).astype(np.float32))

        # Also return arm masks separately (for agnostic image)
        left_arm_mask  = torch.from_numpy((pred == 14).astype(np.float32))
        right_arm_mask = torch.from_numpy((pred == 15).astype(np.float32))
        upper_mask     = torch.from_numpy((pred == 4).astype(np.float32))

        return parse_tensor, upper_mask, left_arm_mask, right_arm_mask

    def make_agnostic(self, person_img, upper_mask, left_arm, right_arm, H=1024, W=768):
        """Erase upper clothing + arms from person image → agnostic image."""
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        person_t = transform(person_img).unsqueeze(0)  # (1,3,H,W)

        # Combined mask of regions to erase
        erase_mask = (upper_mask + left_arm + right_arm).clamp(0, 1)
        erase_mask = erase_mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # Replace erased region with grey (neutral value in normalized space)
        agnostic = person_t * (1 - erase_mask)
        return agnostic  # (1,3,H,W) normalized

    @modal.method()
    def run_tryon(self, person_b64: str, cloth_b64: str) -> dict:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision.transforms as transforms
        import torchgeometry as tgm
        import sys
        sys.path.insert(0, "/root/VITON-HD")
        from utils import gen_noise

        opt = self.opt
        H, W = opt.load_height, opt.load_width  # 1024, 768

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        mask_transform = transforms.ToTensor()

        # ── Decode inputs ──────────────────────────────────────────
        person_img = resize_image(b64_to_pil(person_b64), (W, H))
        cloth_img  = resize_image(b64_to_pil(cloth_b64),  (W, H))
        cloth_mask = generate_cloth_mask(cloth_img)

        # ── Generate real pose ─────────────────────────────────────
        print("Running pose estimation...")
        pose_rgb_img = self.get_pose(person_img, H, W)

        # ── Generate real parse ────────────────────────────────────
        print("Running human parsing...")
        parse_agnostic, upper_mask, left_arm, right_arm = self.get_parse(person_img, H, W)
        parse_agnostic = parse_agnostic.to(self.device)

        # ── Make agnostic image ────────────────────────────────────
        print("Making agnostic image...")
        img_agnostic = self.make_agnostic(
            person_img, upper_mask, left_arm, right_arm, H, W
        ).to(self.device)

        # ── Cloth tensors ──────────────────────────────────────────
        c  = transform(cloth_img).unsqueeze(0).to(self.device)
        cm = mask_transform(cloth_mask).unsqueeze(0).to(self.device)
        pose_rgb_t = transform(pose_rgb_img).unsqueeze(0).to(self.device)

        up    = nn.Upsample(size=(H, W), mode='bilinear')
        gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(self.device)

        with torch.no_grad():
            # ── Part 1: Segmentation (256×192) ────────────────────
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down           = F.interpolate(pose_rgb_t,     size=(256, 192), mode='bilinear')
            c_masked_down       = F.interpolate(c * cm,         size=(256, 192), mode='bilinear')
            cm_down             = F.interpolate(cm,             size=(256, 192), mode='bilinear')
            noise               = gen_noise(cm_down.size()).to(self.device)

            # cm(1)+c_masked(3)+parse_agnostic(13)+pose_rgb(3)+noise(1) = 21ch
            seg_input = torch.cat([
                cm_down, c_masked_down, parse_agnostic_down, pose_down, noise
            ], dim=1)

            parse_pred_down = self.seg_model(seg_input)
            parse_pred      = gauss(up(parse_pred_down))
            parse_pred      = parse_pred.argmax(dim=1)[:, None]

            parse_old = torch.zeros(1, 13, H, W, dtype=torch.float).to(self.device)
            parse_old.scatter_(1, parse_pred, 1.0)

            # Remap 13 → 7 classes
            labels = {
                0: ['background', [0]],
                1: ['paste',      [2, 4, 7, 8, 9, 10, 11]],
                2: ['upper',      [3]],
                3: ['hair',       [1]],
                4: ['left_arm',   [5]],
                5: ['right_arm',  [6]],
                6: ['noise',      [12]],
            }
            parse = torch.zeros(1, 7, H, W, dtype=torch.float).to(self.device)
            for j in range(len(labels)):
                for label in labels[j][1]:
                    parse[:, j] += parse_old[:, label]

            # ── Part 2: GMM (256×192) ──────────────────────────────
            agnostic_gmm    = F.interpolate(img_agnostic,  size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm        = F.interpolate(pose_rgb_t,    size=(256, 192), mode='nearest')
            c_gmm           = F.interpolate(c,             size=(256, 192), mode='nearest')

            # parse_cloth(1)+pose(3)+agnostic(3) = 7ch
            gmm_input = torch.cat([parse_cloth_gmm, pose_gmm, agnostic_gmm], dim=1)
            _, warped_grid = self.gmm_model(gmm_input, c_gmm)
            warped_c  = F.grid_sample(c,  warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            # ── Part 3: ALIAS (full resolution) ───────────────────
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0

            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask

            # agnostic(3)+pose(3)+warped_c(3) = 9ch
            alias_input = torch.cat([img_agnostic, pose_rgb_t, warped_c], dim=1)
            output = self.alias_model(alias_input, parse, parse_div, misalign_mask)

        result_np = output.squeeze(0).cpu()
        result_np = (result_np * 0.5 + 0.5).clamp(0, 1)
        result_np = (result_np.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        return {
            "success": True,
            "result_image": pil_to_b64(Image.fromarray(result_np))
        }


# ── FastAPI ───────────────────────────────────────────────────────
web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TryOnRequest(BaseModel):
    person_image: str
    cloth_image: str


@app.function(
    image=viton_image,
    gpu="T4",
    volumes={"/weights": volume},
    timeout=600,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    return web_app


@web_app.get("/health")
async def health():
    return {"status": "ok"}


@web_app.post("/tryon")
async def tryon(request: TryOnRequest):
    try:
        model = VitonHDModel()
        result = await model.run_tryon.remote.aio(
            request.person_image,
            request.cloth_image
        )
        return {"success": True, "result_image": result["result_image"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))