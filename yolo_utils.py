from __future__ import annotations

from typing import Dict, Any, List


# Path B  YOLO (simple)
# using ultralytics yolov8n (fast + small)


def run_yolo(image_path: str, conf_thres: float = 0.25, max_det: int = 10) -> Dict[str, Any]:
    """
    run yolo on 1 image and return a compact dict:
    - list of dets w label/conf/box
    - counts per label
    """
    # import here so the rest of pipeline still works even if yolo not installed
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # auto download if missing

    results = model.predict(
        source=image_path,
        conf=conf_thres,
        verbose=False,
        imgsz=640,
        max_det=max_det,
    )

    if not results:
        return {"detections": [], "label_counts": {}, "model": "yolov8n.pt"}

    r0 = results[0]
    names = r0.names  # id -> name
    boxes = r0.boxes

    dets: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    if boxes is not None and boxes.xyxy is not None:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy()

        for i in range(len(xyxy)):
            cls_id = int(clss[i])
            label = names.get(cls_id, str(cls_id))
            conf = float(confs[i])
            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]

            dets.append(
                {
                    "label": label,
                    "confidence": round(conf, 6),
                    "box_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )

            counts[label] = counts.get(label, 0) + 1

    return {
        "detections": dets,
        "label_counts": counts,
        "model": "yolov8n.pt",
        "conf_thres": conf_thres,
        "max_det": max_det,
    }

