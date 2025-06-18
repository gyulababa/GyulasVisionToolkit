# scr/coreclasses/filtering/BoxDeduplicator.py

class BoxDeduplicator:
    def __init__(self, iou_threshold=0.3, overlap_threshold=0.7, size_ratio_threshold=2.0):
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
        self.size_ratio_threshold = size_ratio_threshold

    def is_duplicate(self, boxA, boxB):
        iou_val = self._iou(boxA, boxB)
        areaA = self._area(boxA)
        areaB = self._area(boxB)
        size_ratio = max(areaA, areaB) / min(areaA, areaB)

        print(f"[Deduplicator] IOU: {iou_val:.3f} | Size ratio: {size_ratio:.2f}")

        if size_ratio > self.size_ratio_threshold:
            overlap_val = self._relative_overlap(boxA, boxB)
            print(f"[Deduplicator] Relative overlap: {overlap_val:.3f}")
            if overlap_val > self.overlap_threshold:
                return True
        else:
            if iou_val > self.iou_threshold:
                return True
        return False

    def _area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def _intersection(self, boxA, boxB):
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _iou(self, boxA, boxB):
        inter_area = self._intersection(boxA, boxB)
        if inter_area == 0: return 0.0
        areaA = self._area(boxA)
        areaB = self._area(boxB)
        return inter_area / (areaA + areaB - inter_area)

    def _relative_overlap(self, boxA, boxB):
        inter_area = self._intersection(boxA, boxB)
        areaB = self._area(boxB)
        return inter_area / areaB if areaB else 0.0
