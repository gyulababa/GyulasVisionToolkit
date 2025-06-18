# coreclasses/config_loader.py

import yaml
import os

class ConfigLoader:

    def __init__(self, main_config_path, base_path="configs/"):
        self.base_path = base_path
        self.main_config_path = main_config_path

        self.main_cfg = self._load_yaml(os.path.join(base_path, "main", main_config_path))
        self.dedup_cfg = self._load_yaml(os.path.join(base_path, "dedup", "dedup_presets.yaml"))
        self.model_cfg = self._load_yaml(os.path.join(base_path, "model", "model_paths.yaml"))

        self.config = self._merge_configs()

    def _load_yaml(self, path):
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Failed to load YAML from {path}: {e}")
            return {}

    def _merge_configs(self):
        cfg = {}

        # pipeline basic configs directly from main config
        cfg['pipeline'] = self.main_cfg.get("pipeline", {})
        cfg['analyzer'] = self.main_cfg.get("analyzer", {})
        cfg['post_crop_filter'] = self.main_cfg.get("post_crop_filter", {})
        cfg['detector_filter'] = self.main_cfg.get("detector_filter", {})
        cfg['model_selection'] = self.main_cfg.get("model_selection", {})

        # Deduplication handling (resolve presets + overrides)
        dedup = self.main_cfg.get("deduplication", {})
        resolved_dedup = {}

        for target in ["face", "person"]:
            preset_name = dedup.get(f"{target}_preset", "default")
            preset = self.dedup_cfg.get('presets', {}).get(preset_name, {})

            for param in ["iou_threshold", "overlap_threshold", "size_ratio_threshold"]:
                value = dedup.get(f"{target}_{param}")
                resolved_dedup[f"{target}_{param}"] = value if value is not None else preset.get(param, 0.0)

        cfg['deduplication'] = resolved_dedup

        # Resolved model paths from models config
        model_sel = cfg['model_selection']

        # Resolve person_detector_model → yolo model file path
        person_key = model_sel.get("person_detector_model", "default")
        cfg['person_model_path'] = self.model_cfg.get('yolo_models', {}).get(person_key, "")

        # Resolve backend (string, not file)
        backend_key = model_sel.get("face_detector_backend", "default")
        cfg['face_detector_backend'] = self.model_cfg.get('deepface_backends', {}).get(backend_key, "opencv")

        return cfg

    def get(self):
        return self.config
