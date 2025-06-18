# scr/coreclasses/detectors/FaceDetector.py

# ========================================
# Face Detector (DeepFace + hybrid dedup)
# ========================================
from deepface import DeepFace
from scr.coreclasses.filtering.boxdeduplicator import BoxDeduplicator

class FaceDetector:
    def __init__(self, backend="opencv", min_face_size=40, max_face_size=1024,
                 min_aspect_ratio=0.5, max_aspect_ratio=2.0,
                 iou_threshold=0.3, overlap_threshold=0.7, size_ratio_threshold=2.0):
        self.backend = backend
        self.min_face_size = min_face_size
        self.max_face_size = max_face_size
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        self.deduplicator = BoxDeduplicator(iou_threshold, overlap_threshold, size_ratio_threshold)

    def detect_faces(self, image):
        detections = DeepFace.extract_faces(
            img_path=image,
            detector_backend=self.backend,
            enforce_detection=False,
            align=False
        )

        raw_boxes = []
        h, w = image.shape[:2]
        for face in detections:
            region = face['facial_area']
            x1 = max(0, int(region['x']))
            y1 = max(0, int(region['y']))
            x2 = min(int(region['x'] + region['w']), w)
            y2 = min(int(region['y'] + region['h']), h)

            box_w = x2 - x1
            box_h = y2 - y1
            aspect_ratio = box_w / box_h if box_h > 0 else 0

            if (self.min_face_size <= box_w <= self.max_face_size and
                self.min_face_size <= box_h <= self.max_face_size and
                self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                raw_boxes.append(((x1, y1, x2, y2), box_w * box_h))
            else:
                print(f"⚠️ Skipped face box: size={box_w}x{box_h}, aspect={aspect_ratio:.2f}")

        raw_boxes.sort(key=lambda b: b[1])  # smallest area first

        boxes = []
        for (x1, y1, x2, y2), _ in raw_boxes:
            duplicate = False
            for prev_box in boxes:
                if self.deduplicator.is_duplicate((x1, y1, x2, y2), prev_box):
                    duplicate = True
                    print("⚠️ Face duplicate skipped by hybrid check")
                    break
            if not duplicate:
                boxes.append((x1, y1, x2, y2))

        return boxes
