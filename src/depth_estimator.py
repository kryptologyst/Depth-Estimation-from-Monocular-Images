"""
Core depth estimation module using MiDaS models.

This module provides a modern, robust implementation of depth estimation
from monocular images using Intel's MiDaS models with enhanced features.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    A modern depth estimation class using MiDaS models.
    
    This class provides a robust interface for depth estimation from monocular
    images with support for multiple MiDaS model variants, enhanced visualization,
    and comprehensive error handling.
    
    Attributes:
        model: The loaded MiDaS model
        device: The computation device (CPU/GPU)
        model_name: Name of the loaded model
        transforms: Preprocessing transforms for the model
    """
    
    # Available MiDaS models
    AVAILABLE_MODELS = {
        "small": "MiDaS_small",
        "medium": "MiDaS",
        "large": "MiDaS_large",
        "hybrid": "DPT_Hybrid",
        "large_hybrid": "DPT_Large"
    }
    
    def __init__(
        self, 
        model_name: str = "small",
        device: Optional[str] = None,
        force_reload: bool = False
    ):
        """
        Initialize the depth estimator.
        
        Args:
            model_name: Name of the MiDaS model to use. Options: 
                       'small', 'medium', 'large', 'hybrid', 'large_hybrid'
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            force_reload: Whether to force reload the model from torch.hub
            
        Raises:
            ValueError: If model_name is not supported
            RuntimeError: If model loading fails
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Available models: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = None
        self.transforms = None
        
        logger.info(f"Initializing DepthEstimator with model: {model_name}")
        self._load_model(force_reload)
        
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup computation device."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device_obj = torch.device(device)
        logger.info(f"Using device: {device_obj}")
        return device_obj
    
    def _load_model(self, force_reload: bool = False) -> None:
        """Load the MiDaS model and transforms."""
        try:
            model_key = self.AVAILABLE_MODELS[self.model_name]
            
            logger.info(f"Loading MiDaS model: {model_key}")
            self.model = torch.hub.load(
                "intel-isl/MiDaS", 
                model_key, 
                force_reload=force_reload
            )
            self.model.eval()
            self.model.to(self.device)
            
            # Load appropriate transforms
            self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            # Select transform based on model
            if self.model_name in ["small", "medium"]:
                self.transform = self.transforms.small_transform
            elif self.model_name == "large":
                self.transform = self.transforms.default_transform
            else:  # hybrid models
                self.transform = self.transforms.dpt_transform
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def load_image(
        self, 
        image_path: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess an image for depth estimation.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (original_image_rgb, preprocessed_tensor)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded or processed
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load image
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            input_tensor = self.transform(img_rgb).to(self.device)
            
            logger.info(f"Image loaded successfully: {image_path}")
            return img_rgb, input_tensor
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise ValueError(f"Image loading failed: {e}")
    
    def estimate_depth(
        self, 
        input_tensor: torch.Tensor,
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Estimate depth map from input tensor.
        
        Args:
            input_tensor: Preprocessed input tensor
            original_shape: Original image shape (height, width)
            
        Returns:
            Normalized depth map as numpy array
        """
        try:
            with torch.no_grad():
                # Ensure input tensor has correct shape
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
                elif input_tensor.dim() == 4:
                    pass  # Already has batch dimension
                else:
                    raise ValueError(f"Unexpected input tensor shape: {input_tensor.shape}")
                
                # Run inference
                prediction = self.model(input_tensor)
                
                # Handle different prediction shapes
                if prediction.dim() == 2:
                    prediction = prediction.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                elif prediction.dim() == 3:
                    prediction = prediction.unsqueeze(1)  # Add channel dimension
                elif prediction.dim() == 4:
                    pass  # Already correct shape
                else:
                    raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
                
                # Interpolate to original image size
                prediction = F.interpolate(
                    prediction,
                    size=original_shape,
                    mode="bicubic",
                    align_corners=False
                ).squeeze()
                
                # Convert to numpy and normalize
                depth_map = prediction.cpu().numpy()
                depth_map = self._normalize_depth(depth_map)
                
                logger.info("Depth estimation completed successfully")
                return depth_map
                
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            raise RuntimeError(f"Depth estimation failed: {e}")
    
    def _normalize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Normalize depth map to 0-1 range.
        
        Args:
            depth_map: Raw depth map
            
        Returns:
            Normalized depth map
        """
        # Handle edge case where all values are the same
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min < 1e-6:
            return np.zeros_like(depth_map)
        
        return (depth_map - depth_min) / (depth_max - depth_min)
    
    def process_image(
        self, 
        image_path: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline: load image and estimate depth.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (original_image_rgb, depth_map)
        """
        img_rgb, input_tensor = self.load_image(image_path)
        depth_map = self.estimate_depth(input_tensor, img_rgb.shape[:2])
        
        return img_rgb, depth_map
    
    def visualize_results(
        self,
        original_image: np.ndarray,
        depth_map: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (15, 6),
        colormap: str = "inferno"
    ) -> None:
        """
        Visualize original image and depth map.
        
        Args:
            original_image: Original RGB image
            depth_map: Estimated depth map
            save_path: Optional path to save the visualization
            figsize: Figure size for matplotlib
            colormap: Colormap for depth visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Depth map
        im = axes[1].imshow(depth_map, cmap=colormap)
        axes[1].set_title("Depth Map", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay visualization
        overlay = self._create_overlay(original_image, depth_map)
        axes[2].imshow(overlay)
        axes[2].set_title("Depth Overlay", fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def _create_overlay(
        self, 
        original_image: np.ndarray, 
        depth_map: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Create depth overlay on original image.
        
        Args:
            original_image: Original RGB image
            depth_map: Depth map
            alpha: Transparency factor for overlay
            
        Returns:
            Overlay image
        """
        # Convert depth map to RGB using inferno colormap
        cmap = plt.cm.inferno
        depth_rgb = cmap(depth_map)[:, :, :3]
        
        # Blend with original image
        overlay = (1 - alpha) * original_image + alpha * depth_rgb * 255
        return overlay.astype(np.uint8)
    
    def get_depth_statistics(self, depth_map: np.ndarray) -> Dict[str, float]:
        """
        Calculate depth map statistics.
        
        Args:
            depth_map: Depth map
            
        Returns:
            Dictionary with depth statistics
        """
        return {
            "mean_depth": float(np.mean(depth_map)),
            "std_depth": float(np.std(depth_map)),
            "min_depth": float(np.min(depth_map)),
            "max_depth": float(np.max(depth_map)),
            "median_depth": float(np.median(depth_map)),
            "depth_range": float(np.max(depth_map) - np.min(depth_map))
        }
    
    def save_depth_map(
        self, 
        depth_map: np.ndarray, 
        save_path: Union[str, Path],
        format: str = "png"
    ) -> None:
        """
        Save depth map to file.
        
        Args:
            depth_map: Depth map to save
            save_path: Path to save the depth map
            format: File format ('png', 'npy', 'tiff')
        """
        save_path = Path(save_path)
        
        if format.lower() == "npy":
            np.save(save_path, depth_map)
        elif format.lower() == "tiff":
            # Scale to 16-bit for TIFF
            depth_scaled = (depth_map * 65535).astype(np.uint16)
            cv2.imwrite(str(save_path), depth_scaled)
        else:  # PNG
            # Scale to 8-bit for PNG
            depth_scaled = (depth_map * 255).astype(np.uint8)
            cv2.imwrite(str(save_path), depth_scaled)
        
        logger.info(f"Depth map saved to: {save_path}")


def create_sample_image(
    output_path: Union[str, Path],
    size: Tuple[int, int] = (640, 480),
    pattern: str = "gradient"
) -> None:
    """
    Create a sample image for testing.
    
    Args:
        output_path: Path to save the sample image
        size: Image size (width, height)
        pattern: Pattern type ('gradient', 'circles', 'checkerboard')
    """
    output_path = Path(output_path)
    
    if pattern == "gradient":
        # Create a gradient image
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for i in range(size[0]):
            img[:, i] = [int(255 * i / size[0]), 100, int(255 * (1 - i / size[0]))]
    
    elif pattern == "circles":
        # Create circles pattern
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        center_x, center_y = size[0] // 2, size[1] // 2
        for r in range(50, min(center_x, center_y), 50):
            cv2.circle(img, (center_x, center_y), r, (255, 255, 255), 2)
    
    else:  # checkerboard
        # Create checkerboard pattern
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        square_size = 50
        for y in range(0, size[1], square_size):
            for x in range(0, size[0], square_size):
                if (x // square_size + y // square_size) % 2 == 0:
                    img[y:y+square_size, x:x+square_size] = [255, 255, 255]
    
    cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    logger.info(f"Sample image created: {output_path}")


if __name__ == "__main__":
    # Example usage
    estimator = DepthEstimator(model_name="small")
    
    # Create sample image if it doesn't exist
    sample_path = Path("data/sample_image.jpg")
    sample_path.parent.mkdir(exist_ok=True)
    
    if not sample_path.exists():
        create_sample_image(sample_path)
    
    # Process image
    try:
        original_img, depth_map = estimator.process_image(sample_path)
        
        # Visualize results
        estimator.visualize_results(original_img, depth_map)
        
        # Print statistics
        stats = estimator.get_depth_statistics(depth_map)
        print("Depth Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")
        
        # Save depth map
        estimator.save_depth_map(depth_map, "data/depth_map.png")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
