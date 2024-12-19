import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
import argparse
from PIL import Image
from torchvision import transforms

from config import config
from models.dpn_plus import DPNPlus
from utils import load_checkpoint

class FaceRecognitionInference:
    def __init__(self, model_path: Path):
        # Initialize model
        self.model = DPNPlus(num_classes=config.num_classes)
        self.model, _, _, _ = load_checkpoint(
            self.model, None, model_path
        )
        self.model = self.model.to(config.device)
        self.model.eval()

        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std)
        ])

    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        """Preprocess a single image"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.unsqueeze(0)

    def extract_features(self, image: torch.Tensor) -> np.ndarray:
        """Extract features from preprocessed image"""
        with torch.no_grad():
            image = image.to(config.device)
            _, features = self.model(image)
            return features.cpu().numpy()

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors"""
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

    def verify_faces(self, image1_path: Path, image2_path: Path, 
                    threshold: float = 0.6) -> Tuple[bool, float]:
        """Verify if two face images belong to the same person"""
        # Preprocess images
        img1 = self.preprocess_image(image1_path)
        img2 = self.preprocess_image(image2_path)

        # Extract features
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)

        # Compute similarity
        similarity = self.compute_similarity(feat1[0], feat2[0])

        return similarity > threshold, similarity

    def identify_face(self, image_path: Path, 
                     gallery_features: np.ndarray,
                     gallery_labels: List[str]) -> Tuple[str, float]:
        """Identify a face against a gallery of known faces"""
        # Preprocess and extract features
        img = self.preprocess_image(image_path)
        feat = self.extract_features(img)

        # Compute similarities with gallery
        similarities = [self.compute_similarity(feat[0], gf) for gf in gallery_features]
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        return gallery_labels[best_idx], best_similarity

def main():
    parser = argparse.ArgumentParser(description='Face Recognition Testing')
    parser.add_argument('--mode', choices=['verify', 'identify'], required=True,
                        help='Operation mode: verify two faces or identify against gallery')
    parser.add_argument('--image1', type=Path, required=True,
                        help='Path to first image')
    parser.add_argument('--image2', type=Path,
                        help='Path to second image (for verification)')
    parser.add_argument('--gallery_path', type=Path,
                        help='Path to gallery directory (for identification)')
    parser.add_argument('--model_path', type=Path, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Similarity threshold for verification')
    
    args = parser.parse_args()
    
    # Initialize inference
    face_recognition = FaceRecognitionInference(args.model_path)
    
    if args.mode == 'verify':
        if args.image2 is None:
            raise ValueError("Second image path required for verification mode")
            
        is_match, similarity = face_recognition.verify_faces(
            args.image1, args.image2, args.threshold
        )
        
        print(f"Similarity score: {similarity:.4f}")
        print(f"Match: {is_match}")
        
    else:  # identify mode
        if args.gallery_path is None:
            raise ValueError("Gallery path required for identification mode")
            
        # Here you would implement gallery feature extraction
        # This is just a placeholder for the gallery creation logic
        gallery_features = []
        gallery_labels = []
        
        identity, confidence = face_recognition.identify_face(
            args.image1, gallery_features, gallery_labels
        )
        
        print(f"Identified as: {identity}")
        print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()