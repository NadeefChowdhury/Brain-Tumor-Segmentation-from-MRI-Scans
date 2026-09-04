from ultralytics import YOLO



if __name__ == "__main__":
    model = YOLO('yolo26n-seg.pt')  # load a pretrained model (recommended for training)
    model.train(data="config.yaml", epochs=150, imgsz=640)