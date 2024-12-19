import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import Tuple, Optional, List

class FaceAugmentation:
    """Advanced augmentation techniques specific to face recognition"""
    
    def __init__(self, 
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 blur_radius_range: Tuple[float, float] = (0, 1.0),
                 noise_factor: float = 0.05,
                 rotation_range: Tuple[float, float] = (-10, 10),
                 scale_range: Tuple[float, float] = (0.9, 1.1)):
        
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_radius_range = blur_radius_range
        self.noise_factor = noise_factor
        self.rotation_range = rotation_range
        self.scale_range = scale_range

    def apply_brightness(self, image: Image.Image) -> Image.Image:
        """Adjust image brightness"""
        factor = random.uniform(*self.brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def apply_contrast(self, image: Image.Image) -> Image.Image:
        """Adjust image contrast"""
        factor = random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def apply_blur(self, image: Image.Image) -> Image.Image:
        """Apply Gaussian blur"""
        radius = random.uniform(*self.blur_radius_range)
        if radius > 0:
            return image.filter(ImageFilter.GaussianBlur(radius))
        return image

    def apply_noise(self, image: Image.Image) -> Image.Image:
        """Add random noise"""
        img_array = np.array(image)
        noise = np.random.normal(0, self.noise_factor * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def apply_rotation(self, image: Image.Image) -> Image.Image:
        """Rotate image"""
        angle = random.uniform(*self.rotation_range)
        return image.rotate(angle, Image.BILINEAR, expand=False)

    def apply_scale(self, image: Image.Image) -> Image.Image:
        """Scale image"""
        scale = random.uniform(*self.scale_range)
        new_size = tuple(int(dim * scale) for dim in image.size)
        return image.resize(new_size, Image.BILINEAR)

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations"""
        augmentations = [
            self.apply_brightness,
            self.apply_contrast,
            self.apply_blur,
            self.apply_noise,
            self.apply_rotation,
            self.apply_scale
        ]
        
        # Randomly apply 2-4 augmentations
        num_augments = random.randint(2, 4)
        selected_augments = random.sample(augmentations, num_augments)
        
        for augment in selected_augments:
            image = augment(image)
        
        return image

class CutMix:
    """Implementation of CutMix augmentation"""
    
    def __init__(self, 
                 prob: float = 0.5,
                 beta: float = 1.0):
        self.prob = prob
        self.beta = beta

    def get_rand_box(self, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Generate random bounding box"""
        W, H = size
        
        # Sample random width and height
        cut_rat = np.sqrt(1.0 - np.random.beta(self.beta, self.beta))
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Sample random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Get bounding box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

    def __call__(self, 
                 images: torch.Tensor,
                 labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation"""
        
        if random.random() > self.prob:
            return images, labels, 1.0
        
        # Generate random permutation
        rand_idx = torch.randperm(images.size(0))
        
        # Get random bounding box
        bbx1, bby1, bbx2, bby2 = self.get_rand_box(images.size()[2:])
        
        # Compute mixing ratio
        mix_ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / 
                        (images.size()[-1] * images.size()[-2]))
        
        # Create mixed images
        mixed_images = images.clone()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = \
            images[rand_idx, :, bbx1:bbx2, bby1:bby2]
        
        return mixed_images, labels, mix_ratio