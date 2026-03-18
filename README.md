# VTON

A virtual try-on system repository (VTON) with backend and frontend components for cloth warping and rendering.

## Repository structure

- `backend/` - Python model inference and preprocessing.
- `frontend/` - Next.js web UI.
- `checkpoints/` - Pretrained model weights.
- `VITON-HD/`, `VTON-TRYON/` - additional implementation variants and experiments.

## Getting started

### Prerequisites

- Python 3.8+ (for backend)
- Node.js 16+ (for frontend)
- Git

### 1) Install backend dependencies

```bash
cd backend
python -m pip install -r requirements.txt
```

### 2) Install frontend dependencies

```bash
cd frontend
npm install
```

### 3) Download checkpoints

Ensure `checkpoints/` contains `alias_final.pth`, `gmm_final.pth`, `seg_final.pth` (already included in this repo). If missing, obtain from project data source.

### 4) Run backend (sample)

```bash
cd backend
python modal_app.py
```

### 5) Run frontend

```bash
cd frontend
npm run dev
```

## Notes

- There is a root `.gitignore` that ignores Python/Node artifacts, IDE files, OS clutter, and model checkpoint folders.
- If you work in the `VTON-TRYON/` or `VITON-HD/` subtrees, they follow similar structure and may require their own dependency installation.

## License

Specify your license information here (if applicable).