# scr/utils/pipeline_builder.py

import os
from pathlib import Path
from coreclasses.config_loader import ConfigLoader
from coreclasses.managers.modelmanager import ModelManager
from coreclasses.detectors.facedetector import FaceDetector
from coreclasses.detectors.pose_emotion import PoseAndEmotionAnalyzer
from coreclasses.detectors.objectdetector import ObjectDetector

# Absolute path to the configs directory inside the scr package
DEFAULT_CONFIGS_ROOT = Path(__file__).resolve().parent.parent / "configs"

def build_pipeline(config_filename="config_default.yaml", configs_root=DEFAULT_CONFIGS_ROOT):
    # Load config
    cfg_loader = ConfigLoader(config_filename, base_path=configs_root)
    cfg = cfg_loader.get()

    # Initialize model manager to resolve model paths
    model_manager = ModelManager()

    # Set DEEPFACE_HOME env path
    deepface_home_path = os.path.join(model_manager.model_dir, "deepfaceWeights")
    os.environ["DEEPFACE_HOME"] = deepface_home_path
    print(f"✅ DeepFace model path set to: {deepface_home_path}")

    analyzer = PoseAndEmotionAnalyzer(
        enforce_detection=True,
        detector_backend=cfg['face_detector_backend'],
        full_analysis=cfg['pipeline'].get('full_analysis', False),
        enable_validation=cfg['analyzer'].get('enable_validation', False),
        verification_threshold=cfg['analyzer'].get('verification_threshold', 0.4),
        preview=cfg['pipeline'].get('preview', False)
    )

    face_detector = FaceDetector(
        backend=cfg['face_detector_backend'],
        min_face_size=cfg['detector_filter'].get('detector_face_min_size', 40),
        max_face_size=cfg['detector_filter'].get('detector_face_max_size', 1024),
        min_aspect_ratio=cfg['detector_filter'].get('detector_face_min_aspect_ratio', 0.5),
        max_aspect_ratio=cfg['detector_filter'].get('detector_face_max_aspect_ratio', 2.0),
        iou_threshold=cfg['deduplication'].get('face_iou_threshold', 0.3),
        overlap_threshold=cfg['deduplication'].get('face_overlap_threshold', 0.7),
        size_ratio_threshold=cfg['deduplication'].get('face_size_ratio_threshold', 2.0)
    )

    person_detector = ObjectDetector(
        model_path=cfg['person_model_path'],
        confidence=0.3,
        iou_threshold=cfg['deduplication'].get('person_iou_threshold', 0.3),
        overlap_threshold=cfg['deduplication'].get('person_overlap_threshold', 0.7),
        size_ratio_threshold=cfg['deduplication'].get('person_size_ratio_threshold', 2.0)
    )

    return {
        "analyzer": analyzer,
        "face_detector": face_detector,
        "person_detector": person_detector,
        "config": cfg
    }
