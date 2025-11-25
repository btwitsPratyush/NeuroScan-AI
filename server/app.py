# server/app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, base64, os, sys, traceback

import torch
import torch.nn as nn
import numpy as np
import cv2
import scipy.io

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pt")

# LABELS: must match training class_map exactly
classes = ["glioma", "meningioma", "pituitary", "notumor"]

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(classes))
        )

    def forward(self, x):
        return self.classifier(self.features(x))


device = "cpu"
model = SmallCNN().to(device)

# Try loading model (state_dict or full model)
if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location=device)
        # if it's a state_dict
        if isinstance(state, dict) and any(k.startswith('features') or k.startswith('classifier') for k in state.keys()):
            model.load_state_dict(state)
            model.eval()
            print("✅ Model loaded successfully (state_dict).")
        else:
            # try full model
            model = state.to(device)
            model.eval()
            print("✅ Model loaded successfully (full model).")
    except Exception as e:
        print("❌ Model load error:", e)
        traceback.print_exc()
else:
    print("⚠️ Model not found at", MODEL_PATH, "- server will run with untrained model (predictions meaningless).")


def preprocess(img_pil):
    img = img_pil.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)

    return torch.tensor(arr, dtype=torch.float32)


def mat_to_pil(mat_bytes):
    """
    Robustly load a .mat file and return a PIL image.
    """
    data = scipy.io.loadmat(io.BytesIO(mat_bytes))
    best = None
    best_size = 0
    for k, v in data.items():
        try:
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                size = v.size
                if size > best_size:
                    best = v
                    best_size = size
        except Exception:
            continue
    if best is None:
        raise ValueError("No suitable ndarray found inside .mat")
    arr = best.astype("float32")
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = arr - np.min(arr)
    if np.max(arr) > 0:
        arr = arr / np.max(arr)
    arr = (arr * 255).astype("uint8")
    if arr.ndim == 2:
        pil = Image.fromarray(arr).convert("L").convert("RGB")
    elif arr.shape[2] == 1:
        pil = Image.fromarray(arr[:, :, 0]).convert("RGB")
    else:
        pil = Image.fromarray(arr.astype("uint8"))
    return pil


def simple_heatmap(inp_tensor, img_pil):
    activations = None

    def hook(module, inp, out):
        nonlocal activations
        activations = out.detach().cpu().numpy()[0]

    # Try hooking last conv in features
    hook_handle = None
    try:
        # model.features is Sequential [conv,relu,pool,...], hook last element that outputs activation maps
        hook_handle = model.features[-1].register_forward_hook(hook)
    except Exception:
        hook_handle = None

    with torch.no_grad():
        _ = model(inp_tensor.unsqueeze(0))

    if hook_handle is not None:
        try:
            hook_handle.remove()
        except Exception:
            pass

    if activations is None:
        return None

    cam = np.mean(activations, axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    img_np = np.array(img_pil.resize((224, 224)))[:, :, ::-1]
    heatmap = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, 0.4, img_np, 0.6, 0)

    _, impng = cv2.imencode(".png", overlay)
    return base64.b64encode(impng.tobytes()).decode("utf-8")



def is_likely_mri(img_pil):
    # Convert to numpy
    img_arr = np.array(img_pil.convert("RGB"))
    
    # 1. Check for color (saturation)
    hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    mean_sat = np.mean(saturation)
    
    # Threshold: MRIs are grayscale, so saturation should be very low.
    if mean_sat > 25: # 0-255 scale
        return False, f"Image has high color saturation ({mean_sat:.1f}), likely not an MRI."
    
    # 2. Check for bright background (Documents usually have white background)
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    
    # Check corners (MRIs usually have black corners)
    h, w = gray.shape
    corner_size = min(h, w) // 10
    # Avoid zero size
    corner_size = max(1, corner_size)
    
    corners = [
        gray[0:corner_size, 0:corner_size],
        gray[0:corner_size, w-corner_size:w],
        gray[h-corner_size:h, 0:corner_size],
        gray[h-corner_size:h, w-corner_size:w]
    ]
    mean_corner_brightness = np.mean([np.mean(c) for c in corners])
    
    if mean_corner_brightness > 60: # Threshold for corner brightness (MRIs are usually < 20)
        return False, f"Image has bright corners (brightness {mean_corner_brightness:.1f}), likely a document or photo."
        
    # 3. Check overall brightness (MRIs are mostly dark)
    mean_brightness = np.mean(gray)
    if mean_brightness > 120:
        return False, f"Image is too bright overall ({mean_brightness:.1f}), likely a document."

    return True, ""


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    b = await file.read()
    filename = (file.filename or "").lower()
    # log receive
    try:
        print(f"[upload] Received: filename={file.filename} size={len(b)} bytes")
    except Exception:
        pass

    try:
        if filename.endswith(".mat"):
            img = mat_to_pil(b)
        else:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            
            # Validate MRI
            is_valid, reason = is_likely_mri(img)
            if not is_valid:
                raise HTTPException(status_code=400, detail=reason)
                
    except HTTPException as he:
        raise he
    except Exception as e:
        return {"error": "cannot read uploaded file", "detail": str(e)}

    inp = preprocess(img)

    with torch.no_grad():
        logits = model(inp.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    label = classes[idx] if idx < len(classes) else str(idx)
    prob = float(probs[idx])

    heatmap = None
    try:
        heatmap = simple_heatmap(inp, img)
    except Exception as e:
        print("Heatmap error:", e)

    return {
        "label": label,
        "probability": prob,    # float 0..1
        "heatmap": heatmap,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return HTMLResponse(open(index_path, "r", encoding="utf-8").read())
    return HTMLResponse("<html><body><h3>NeuroScan API</h3></body></html>")