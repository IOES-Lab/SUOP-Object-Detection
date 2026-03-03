from ultralytics import YOLO

def train_model(data_path, epochs=300, img_size=416):
    model = YOLO("yolov8s.yaml")
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=4,
        device="cpu",
        workers=2,
        cache=False,
        amp=False,
        val=True,
        augment=False,
        name="new_model_train"
    )

if __name__ == "__main__":
    train_model(
        data_path="/path/to/...",
        epochs=100,
        img_size=416
    )
