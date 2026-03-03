import os
import cv2
from ultralytics import YOLO

def load_model(weights_path):
    return YOLO(weights_path)

def detect_and_show(model, image_path, conf_thres=0.2, iou_thres=0.45):
    results = model.predict(
        source=image_path,
        conf=conf_thres,
        iou=iou_thres,
        save=False
    )
    df = results[0].to_df()
    if df.empty:
        return False
    annotated = results[0].plot()
    cv2.imshow(os.path.basename(image_path), annotated[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyWindow(os.path.basename(image_path))
    return True

def show_only_detections(weights_path, folder_path, conf_thres=0.2, iou_thres=0.45):
    model = load_model(weights_path)
    detected_files = []

    for fn in sorted(os.listdir(folder_path)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(folder_path, fn)
        ok = detect_and_show(model, img_path, conf_thres, iou_thres)
        if ok:
            detected_files.append(fn)

    cv2.destroyAllWindows()
    print("\n=== image list ===")
    for fn in detected_files:
        print(fn)

if __name__ == "__main__":
    weights_path = "/path/to/..."
    image_folder = "/path/to/..."
    show_only_detections(
        weights_path,
        image_folder,
        conf_thres=0.2,
        iou_thres=0.45
    )
