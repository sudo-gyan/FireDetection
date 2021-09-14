import os
from imageai.Detection.Custom import (
    CustomObjectDetection, CustomVideoObjectDetection)


class FireDetector:

    def __init__(self):
        self.img_detector = CustomObjectDetection()
        self.video_detector = CustomVideoObjectDetection()

    def _load_model(self, obj):
        obj.setModelTypeAsYOLOv3()
        obj.setModelPath(
            detection_model_path=os.path.join("models", "detection_model.h5"))
        obj.setJsonPath(
            configuration_json=os.path.join("models", "detection_config.json"))
        obj.loadModel()

    def load_image_detector(self):
        self._load_model(self.img_detector)

    def load_video_detector(self):
        self._load_model(self.video_detector)

    def detect_from_video(self, input_path: str, output_path: str,
                          fps=20, confidence=40, log=True):
        self.video_detector.detectObjectsFromVideo(
            input_file_path=input_path,
            frames_per_second=fps,
            output_file_path=output_path,
            minimum_percentage_probability=confidence,
            log_progress=log)

    def detect_from_image(self, input_path: str, output_path: str,
                          confidence=40):
        self.img_detector.detectObjectsFromImage(
            input_image=input_path,
            output_image_path=output_path,
            minimum_percentage_probability=confidence)

    def detect_from_array(self, image_array, confidence=40):
        return self.img_detector.detectObjectsFromImage(
            input_image=image_array,
            input_type="array",
            output_type="array",
            minimum_percentage_probability=confidence)
