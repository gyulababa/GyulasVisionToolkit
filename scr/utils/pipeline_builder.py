# scr/utils/pipeline_builder.py

import os
import cv2
from scr.coreclasses.config_loader import ConfigLoader
from scr.coreclasses.managers.modelmanager import ModelManager
from scr.coreclasses.detectors.facedetector import FaceDetector
from scr.coreclasses.detectors.pose_emotion import PoseAndEmotionAnalyzer
from scr.coreclasses.detectors.objectdetector import ObjectDetector


class ProcessingPipeline:
    """High level wrapper holding all detectors and analyzers."""

    def __init__(self, config_filename="config_default.yaml", configs_root="configs/"):
        # Load config
        cfg_loader = ConfigLoader(config_filename, base_path=configs_root)
        self.cfg = cfg_loader.get()

        # Initialize model manager to resolve model paths
        model_manager = ModelManager()

        # Set DEEPFACE_HOME env path
        deepface_home_path = os.path.join(model_manager.model_dir, "deepfaceWeights")
        os.environ["DEEPFACE_HOME"] = deepface_home_path
        print(f"✅ DeepFace model path set to: {deepface_home_path}")

        self.analyzer = PoseAndEmotionAnalyzer(
            enforce_detection=True,
            detector_backend=self.cfg['face_detector_backend'],
            full_analysis=self.cfg['pipeline'].get('full_analysis', False),
            enable_validation=self.cfg['analyzer'].get('enable_validation', False),
            verification_threshold=self.cfg['analyzer'].get('verification_threshold', 0.4),
            preview=self.cfg['pipeline'].get('preview', False)
        )

        self.face_detector = FaceDetector(
            backend=self.cfg['face_detector_backend'],
            min_face_size=self.cfg['detector_filter'].get('detector_face_min_size', 40),
            max_face_size=self.cfg['detector_filter'].get('detector_face_max_size', 1024),
            min_aspect_ratio=self.cfg['detector_filter'].get('detector_face_min_aspect_ratio', 0.5),
            max_aspect_ratio=self.cfg['detector_filter'].get('detector_face_max_aspect_ratio', 2.0),
            iou_threshold=self.cfg['deduplication'].get('face_iou_threshold', 0.3),
            overlap_threshold=self.cfg['deduplication'].get('face_overlap_threshold', 0.7),
            size_ratio_threshold=self.cfg['deduplication'].get('face_size_ratio_threshold', 2.0),
        )

        self.person_detector = ObjectDetector(
            model_path=self.cfg['person_model_path'],
            confidence=0.3,
            iou_threshold=self.cfg['deduplication'].get('person_iou_threshold', 0.3),
            overlap_threshold=self.cfg['deduplication'].get('person_overlap_threshold', 0.7),
            size_ratio_threshold=self.cfg['deduplication'].get('person_size_ratio_threshold', 2.0),
        )

    def process(self, image):
        """Run detection and analysis on a single image."""
        draw_person = self.cfg['pipeline'].get('draw_person_box', False)

        person_results = self.person_detector.infer(
            image,
            draw_boxes=draw_person,
            allowed_classes=['person']
        )[0]
        visual_img = person_results['visualized'] if person_results['visualized'] is not None else image.copy()

        face_boxes = self.face_detector.detect_faces(image)

        results = []
        for (x1, y1, x2, y2) in face_boxes:
            face_crop = image[y1:y2, x1:x2]
            analysis = self.analyzer.predict_emotion(face_crop)
            if analysis is None:
                continue

            if isinstance(analysis, dict):
                res = analysis
                res['box'] = (x1, y1, x2, y2)
            else:
                res = {'emotion': analysis, 'box': (x1, y1, x2, y2)}
            results.append(res)

            cv2.rectangle(visual_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = res.get('dominant_emotion', res.get('emotion', ''))
            cv2.putText(visual_img, str(label), (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return results, visual_img

def build_pipeline(config_filename="config_default.yaml", configs_root="configs/"):
    """Return a ready-to-use ProcessingPipeline instance."""
    return ProcessingPipeline(config_filename=config_filename, configs_root=configs_root)
