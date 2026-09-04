import numpy as np
import cv2
from PIL import Image


MASK_COLOR  = (0, 200, 100)   # green overlay for tumor mask
BOX_COLOR   = (255, 60,  60)  # red bounding boxes
BOX_THICK   = 2
MASK_ALPHA  = 0.40            # transparency of mask overlay


def overlay_results(image_pil, boxes, mask):
    """
    Draw YOLO bounding boxes and segmentation mask on the original image.

    Args:
        image_pil : PIL.Image (RGB)
        boxes     : list of [x1, y1, x2, y2]
        mask      : np.uint8 H×W binary mask (0 or 1)

    Returns:
        PIL.Image with overlays
        PIL.Image of the mask alone (grayscale, 0/255)
    """
    img = np.array(image_pil.convert("RGB")).copy()

    # --- Mask overlay ---
    color_mask = np.zeros_like(img)
    color_mask[mask == 1] = MASK_COLOR
    img = cv2.addWeighted(img, 1 - MASK_ALPHA, color_mask, MASK_ALPHA, 0)

    # --- Bounding boxes ---
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICK)
        cv2.putText(
            img, "Tumor", (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, BOX_COLOR, 2, cv2.LINE_AA
        )

    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
    return Image.fromarray(img), mask_img


def tumor_stats(mask):
    """Return basic stats about the predicted tumor region."""
    total_pixels  = mask.size
    tumor_pixels  = int(mask.sum())
    tumor_percent = tumor_pixels / total_pixels * 100
    return {
        "Tumor Detected": "Yes" if tumor_pixels > 0 else "No",
        "Tumor Area (px)": tumor_pixels,
        "Tumor Coverage":  f"{tumor_percent:.2f}%",
    }
