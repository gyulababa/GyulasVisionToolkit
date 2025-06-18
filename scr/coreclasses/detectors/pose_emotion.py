# scr/coreclasses/detectors/pose_emotion.py

import cv2
from deepface import DeepFace

class PoseAndEmotionAnalyzer:
    def __init__(self, enforce_detection=False, detector_backend='opencv', preview=False, full_analysis=False, verification_threshold=0.4, enable_validation=False):
        """
        enforce_detection: if True, DeepFace runs internal detection
        detector_backend: used if enforce_detection is True
        preview: optional live preview window for debugging
        full_analysis: if True, run full DeepFace analysis (age, gender, emotion, race)
        verification_threshold: threshold for self-verification distance (lower = stricter)
        enable_validation: if True, perform self-verification before emotion analysis
        """
        self.enforce_detection = enforce_detection
        self.detector_backend = detector_backend
        self.preview = preview
        self.full_analysis = full_analysis
        self.verification_threshold = verification_threshold
        self.enable_validation = enable_validation
        self.window_name = "Emotion Preview"
        self.window_opened = False

    def predict_emotion(self, face_img):
        """
        face_img: cropped face image (BGR or RGB)
        returns: either simple emotion string or full analysis dict
        """
        self._show_preview(face_img)

        if self.enable_validation:
            if not self.is_valid_face(face_img):
                print("⚠️ Face validation failed. Skipping.")
                return None

        result = DeepFace.analyze(
            img_path=face_img,
            actions=['emotion', 'age', 'gender', 'race'] if self.full_analysis else ['emotion'],
            enforce_detection=self.enforce_detection,
            detector_backend=self.detector_backend
        )

        result = result[0]  # DeepFace returns list

        if self.full_analysis:
            return {
                "dominant_emotion": result.get('dominant_emotion'),
                "emotion": result.get('emotion'),
                "age": result.get('age'),
                "gender": result.get('gender'),
                "race": result.get('race')
            }
        else:
            return result.get('dominant_emotion', 'unknown')

    def is_valid_face(self, face_img):
        """
        Validates face using self-verification distance.
        Returns True if face verification distance <= threshold.
        """
        try:
            distance = self._verify_face(face_img)
            return distance <= self.verification_threshold
        except Exception as e:
            print(f"⚠️ Verification error: {e}")
            return False

    def _verify_face(self, face_img):
        """
        Internal method for self-verification.
        Returns distance score.
        """
        result = DeepFace.verify(
            img1_path=face_img,
            img2_path=face_img,
            detector_backend=self.detector_backend,
            enforce_detection=False
        )
        distance = result['distance']
        return distance

    def _show_preview(self, img):
        if not self.preview:
            return

        if img is not None:
            if img.shape[2] == 3:
                img_bgr = img
            else:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if not self.window_opened:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self.window_opened = True

            cv2.imshow(self.window_name, img_bgr)
            cv2.waitKey(1)

    @staticmethod
    def classify_pose(pose_tuple):
        yaw, pitch, roll = pose_tuple
        if abs(yaw) < 15 and abs(pitch) < 10: return 'frontal'
        if yaw < -15: return 'semi-left'
        if yaw > 15: return 'semi-right'
        if pitch > 10: return 'mild-up'
        if pitch < -10: return 'mild-down'
        return 'frontal'

    def close_preview(self):
        if self.preview and self.window_opened:
            cv2.destroyWindow(self.window_name)
            self.window_opened = False
