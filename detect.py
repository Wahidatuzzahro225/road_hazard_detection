from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(
    weights=ROOT / "best.pt",
    source=1,  #ROOT / "videos/test.mp4" 0=webcam_mac 1=iphone,
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    device="",
    view_img=True,
    save=True,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=True,
):

    # Load model
    model = YOLO(weights)

    # Predict
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        show=view_img,
        save=save,
        project=project,
        name=name,
        exist_ok=exist_ok,
    )

    # Print detection results
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            print(f"Detected: {label}, Confidence: {conf:.2f}")


if __name__ == "__main__":
    run()
