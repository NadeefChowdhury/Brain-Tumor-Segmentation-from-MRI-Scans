from ultralytics import YOLO

import cv2
from ultralytics.utils.plotting import Annotator

model_path = 'best.pt'

image_path = 'test/3.jpg'

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)
annotator = Annotator(img, example=model.names)
print(results[0].boxes)
for result in results:
    for j, mask in enumerate(result.masks.data):
        mask = mask.cpu()
        mask = mask.numpy() * 255

        mask = cv2.resize(mask, (W, H))

        cv2.imshow('Mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
for box in results[0].boxes.xyxy.cpu():
    width, height, area = annotator.get_bbox_dimension(box)
    print(f"Bounding Box Width {width.item()}, Height {height.item()}, Area {area.item()}")