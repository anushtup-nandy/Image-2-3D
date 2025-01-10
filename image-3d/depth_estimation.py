import torch
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from PIL import Image
import numpy as np

class DepthEstimator:
    def __init__(self, model_name="Intel/dpt-large", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
        self.depth_model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)

    def estimate_depth(self, image):
        """Estimate depth map from an image."""
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            depth_map = outputs.predicted_depth.cpu().numpy().squeeze()
        return depth_map