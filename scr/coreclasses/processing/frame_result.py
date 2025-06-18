# src/coreclasses/processing/frame_result.py

class FrameResult:
    def __init__(self, frame_index, timestamp):
        self.frame_index = frame_index
        self.timestamp = timestamp  # seconds or ms depending on video source
        self.detections = []  # list of dicts with 'label', 'box', 'score', etc.
        self.meta = {}  # any extra info

    def add_detection(self, label, box, score=None, **kwargs):
        entry = {
            "label": label,
            "box": box,
            "score": score,
        }
        entry.update(kwargs)
        self.detections.append(entry)

    def get_labels(self):
        return [d['label'] for d in self.detections]

    def get_boxes(self):
        return [d['box'] for d in self.detections]

    def __repr__(self):
        return f"<FrameResult {self.frame_index} @ {self.timestamp}s | {len(self.detections)} detections>"
