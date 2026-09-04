import torch
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
from ultralytics import YOLO

IMG_SIZE = 256
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model builders (must match your Kaggle training code exactly) ────────────

def build_resumet():
    """smp.Unet("resnet34", encoder_weights="imagenet", classes=1) — no activation"""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # weights loaded from .pt
        in_channels=3,
        classes=1,
        # NO activation= here; your model outputs raw logits
    )


def build_att_unet():
    """smp.Unet("resnet34", ..., decoder_attention_type="scse", classes=1)"""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",
    )


def build_unetpp():
    """smp.UnetPlusPlus("resnet34", encoder_weights="imagenet", classes=1)"""
    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )


def load_all_models(yolo_path, resumet_path, att_unet_path, unetpp_path):
    yolo = YOLO(yolo_path)

    resumet = build_resumet()
    resumet.load_state_dict(torch.load(resumet_path, map_location=DEVICE))
    resumet.to(DEVICE).eval()

    att_unet = build_att_unet()
    att_unet.load_state_dict(torch.load(att_unet_path, map_location=DEVICE))
    att_unet.to(DEVICE).eval()

    unetpp = build_unetpp()
    unetpp.load_state_dict(torch.load(unetpp_path, map_location=DEVICE))
    unetpp.to(DEVICE).eval()

    return yolo, resumet, att_unet, unetpp


def preprocess(image_np):
    """
    Matches your Kaggle __getitem__ exactly:
      - BGR→RGB (cv2 reads BGR; we receive RGB PIL images, so no swap needed)
      - resize to 256x256
      - /255.0
      - transpose (H,W,C) → (C,H,W)
      - unsqueeze batch dim
    """
    resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))          # RGB already
    tensor  = torch.tensor(resized / 255.0, dtype=torch.float32)  # (256,256,3)
    tensor  = tensor.permute(2, 0, 1).unsqueeze(0)                # (1,3,256,256)
    return tensor.to(DEVICE)


def run_inference(image_pil, yolo, seg_model):
    """
    Single-model inference.

    Args:
        image_pil : PIL.Image (RGB)
        yolo      : YOLO model
        seg_model : one smp segmentation model (outputs logits)

    Returns:
        boxes         : list of [x1, y1, x2, y2] in original image coords
        mask_resized  : np.uint8 H×W binary mask (0/1), original image size
    """
    orig_np = np.array(image_pil.convert("RGB"))
    orig_h, orig_w = orig_np.shape[:2]

    # YOLO detection
    results = yolo(orig_np)
    boxes = []
    if results[0].boxes is not None and len(results[0].boxes):
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

    # Segmentation — sigmoid AFTER the model (logits output)
    tensor = preprocess(orig_np)
    with torch.no_grad():
        logits = seg_model(tensor)              # (1,1,256,256) logits
        pred   = torch.sigmoid(logits)          # probabilities
        mask_256 = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Resize mask back to original image dimensions
    mask_resized = cv2.resize(
        mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )
    return boxes, mask_resized


def run_ensemble(image_pil, yolo, resumet, att_unet, unetpp):
    """
    Ensemble of all three models — matches your Kaggle ensemble_predict().
    Returns averaged probability mask, thresholded at 0.5.
    """
    orig_np = np.array(image_pil.convert("RGB"))
    orig_h, orig_w = orig_np.shape[:2]

    results = yolo(orig_np)
    boxes = []
    if results[0].boxes is not None and len(results[0].boxes):
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

    tensor = preprocess(orig_np)
    with torch.no_grad():
        p1 = torch.sigmoid(resumet(tensor))
        p2 = torch.sigmoid(att_unet(tensor))
        p3 = torch.sigmoid(unetpp(tensor))
        avg = (p1 + p2 + p3) / 3

    mask_256 = (avg.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(
        mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )
    return boxes, mask_resized
