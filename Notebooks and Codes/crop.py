from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm

# Load trained YOLO model
model = YOLO("best.pt")

# Paths
image_dir = "images/"
mask_dir = "masks/"   # if you have masks
out_img = "cropped/images/"
out_mask = "cropped/masks/"

os.makedirs(out_img, exist_ok=True)
os.makedirs(out_mask, exist_ok=True)

for img_name in tqdm(os.listdir(image_dir)):
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)

    results = model(img)[0]

    for i, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)

        crop = img[y1:y2, x1:x2]
        cv2.imwrite(f"{out_img}/{img_name}_{i}.png", crop)

        # Crop mask if exists
        mask_path = os.path.join(mask_dir, img_name)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0)
            crop_mask = mask[y1:y2, x1:x2]
            cv2.imwrite(f"{out_mask}/{img_name}_{i}.png", crop_mask)