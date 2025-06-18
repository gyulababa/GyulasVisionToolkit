# scr/coreclasses/detectors/ObjectDetector.py
# ========================================
# Object Detector (YOLO person detector)
# ========================================
import os
import numpy as np
import cv2
from ultralytics import YOLO
from coreclasses.managers.modelmanager import ModelManager
from coreclasses.filtering.boxdeduplicator import BoxDeduplicator

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", device="cpu", confidence=0.3,
                 iou_threshold=0.3, overlap_threshold=0.7, size_ratio_threshold=2.0):
        self.model_path = model_path
        self.device = device
        self.confidence = confidence

        self.deduplicator = BoxDeduplicator(iou_threshold, overlap_threshold, size_ratio_threshold)

        model_manager = ModelManager("models")
        resolved_path = model_manager.load_model(self.model_path)

        self.model = YOLO(resolved_path)
        self.model.to(self.device)

        self.class_names = self.model.names if hasattr(self.model, 'names') else None

    def infer(self, inputs, slack=0.0, draw_boxes=False, allowed_classes=None):
        if not isinstance(inputs, list):
            inputs = [inputs]

        results = self.model(inputs, verbose=False, device=self.device)
        detections = []

        for idx, r in enumerate(results):
            img = self._load_image(inputs[idx])
            img_h, img_w = img.shape[:2]
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()

            raw_boxes = []
            for box, conf, cls_id in zip(boxes, confs, clss):
                if conf < self.confidence:
                    continue

                class_name = self.class_names[int(cls_id)] if self.class_names else str(cls_id)
                if allowed_classes and class_name not in allowed_classes:
                    continue

                x1, y1, x2, y2 = box.astype(int)
                x1, y1, x2, y2 = self._expand_box(x1, y1, x2, y2, img_w, img_h, slack)

                raw_boxes.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'box': (x1, y1, x2, y2)
                })

            filtered = self._deduplicate(raw_boxes)

            detections.append({
                'input': inputs[idx],
                'detections': filtered,
                'visualized': self._draw_boxes(img, filtered) if draw_boxes else None
            })

        return detections

    def _deduplicate(self, detections):
        filtered = []
        for det in sorted(detections, key=lambda d: -d['confidence']):
            duplicate = False
            for prev in filtered:
                if self.deduplicator.is_duplicate(det['box'], prev['box']):
                    duplicate = True
                    print("⚠️ Object duplicate skipped by hybrid check")
                    break
            if not duplicate:
                filtered.append(det)
        return filtered

    def _load_image(self, img_input):
        if isinstance(img_input, np.ndarray):
            return img_input
        img = cv2.imread(img_input)
        if img is None:
            raise ValueError(f"Failed to read image: {img_input}")
        return img

    def _expand_box(self, x1, y1, x2, y2, img_w, img_h, slack):
        bw, bh = x2 - x1, y2 - y1
        pad_w, pad_h = int(bw * slack / 2), int(bh * slack / 2)
        nx1 = max(0, x1 - pad_w)
        ny1 = max(0, y1 - pad_h)
        nx2 = min(img_w, x2 + pad_w)
        ny2 = min(img_h, y2 + pad_h)
        return nx1, ny1, nx2, ny2

    def _draw_boxes(self, img, boxes):
        for det in boxes:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{det['class']} {det['confidence']:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img
