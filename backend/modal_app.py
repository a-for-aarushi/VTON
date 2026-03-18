import modal
import io
import base64
import tempfile
import os
from PIL import Image

app = modal.App("idm-vton")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install([
        "Pillow",
        "numpy<2",
        "fastapi",
        "python-multipart",
        "pydantic",
        "gradio-client>=1.0.0",
        "httpx",
    ])
)


def b64_to_pil(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@app.cls(
    image=image,
    gpu="A10G",
    timeout=300,
    scaledown_window=300,
)
class IDMVTONModel:

    @modal.enter()
    def load_model(self):
        from gradio_client import Client
        # Use a duplicate space that's less loaded
        try:
            self.client = Client("yisol/IDM-VTON", verbose=False)
            print("✅ Connected to IDM-VTON!")
        except Exception as e:
            print(f"Primary space failed: {e}, trying backup...")
            self.client = Client("multimodalart/IDM-VTON", verbose=False)
            print("✅ Connected to backup IDM-VTON!")

    @modal.method()
    def run_tryon(self, person_b64: str, cloth_b64: str) -> dict:
        from gradio_client import handle_file

        person_img = b64_to_pil(person_b64)
        cloth_img  = b64_to_pil(cloth_b64)

        # Resize to expected dimensions
        person_img = person_img.resize((768, 1024), Image.LANCZOS)
        cloth_img  = cloth_img.resize((768, 1024), Image.LANCZOS)

        person_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        person_img.save(person_tmp.name)
        person_tmp.close()

        cloth_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cloth_img.save(cloth_tmp.name)
        cloth_tmp.close()

        try:
            print(f"Calling IDM-VTON with person: {person_tmp.name}, cloth: {cloth_tmp.name}")

            result = self.client.predict(
                dict={
                    "background": handle_file(person_tmp.name),
                    "layers": [],
                    "composite": None
                },
                garm_img=handle_file(cloth_tmp.name),
                garment_des="upper body clothing",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )

            print(f"Result: {result}")
            result_img = Image.open(result[0]).convert("RGB")

            return {
                "success": True,
                "result_image": pil_to_b64(result_img)
            }
        finally:
            os.unlink(person_tmp.name)
            os.unlink(cloth_tmp.name)


# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    image=image,
    gpu="A10G",
    timeout=300,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    return web_app


@web_app.get("/health")
async def health():
    return {"status": "ok", "model": "IDM-VTON"}


@web_app.post("/tryon")
async def tryon(request: TryOnRequest):
    try:
        model = IDMVTONModel()
        result = await model.run_tryon.remote.aio(
            request.person_image,
            request.cloth_image
        )
        return {"success": True, "result_image": result["result_image"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))