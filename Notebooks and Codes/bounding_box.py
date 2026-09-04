import cv2
from ultralytics import YOLO

# 1. Load your trained custom segmentation model (e.g., 'best.pt' or 'yolo11n-seg.pt')
model = YOLO("best.pt")

# 2. Run prediction on your local image file
# boxes=True ensures bounding boxes are shown (this is True by default)
results = model.predict(source="20.png", boxes=True)

# 3. Extract the visual results from the first prediction image
annotated_img = results[0].plot(masks=False)
cv2.imwrite("output_bbox_only.jpg", annotated_img)

print("Image successfully saved as 'output_bbox_only.jpg'")
# 4. Display the output image using OpenCV
cv2.imshow("YOLO Segmentation & Bounding Boxes", annotated_img)

# 5. Wait for any key press to close the image window safely
cv2.waitKey(0)
cv2.destroyAllWindows()