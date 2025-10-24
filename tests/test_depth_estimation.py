"""
Test suite for depth estimation project.

This module contains comprehensive unit tests and integration tests
for the depth estimation functionality.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import cv2
from unittest.mock import patch, MagicMock

from src.depth_estimator import DepthEstimator, create_sample_image
from src.config import ConfigManager, AppConfig, ModelConfig


class TestDepthEstimator(unittest.TestCase):
    """Test cases for DepthEstimator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_image_path = self.temp_dir / "test_image.jpg"
        
        # Create a test image
        create_sample_image(self.sample_image_path, size=(320, 240))
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.depth_estimator.torch.hub.load')
    def test_initialization(self, mock_hub_load):
        """Test DepthEstimator initialization."""
        # Mock the model and transforms
        mock_model = MagicMock()
        mock_transforms = MagicMock()
        mock_transform = MagicMock()
        
        mock_hub_load.side_effect = [mock_model, mock_transforms]
        mock_transforms.small_transform = mock_transform
        
        # Initialize estimator
        estimator = DepthEstimator(model_name="small")
        
        # Verify initialization
        self.assertEqual(estimator.model_name, "small")
        self.assertIsNotNone(estimator.device)
        self.assertEqual(estimator.model, mock_model)
        self.assertEqual(estimator.transform, mock_transform)
        
        # Verify model setup
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once()
    
    def test_invalid_model_name(self):
        """Test initialization with invalid model name."""
        with self.assertRaises(ValueError):
            DepthEstimator(model_name="invalid_model")
    
    def test_load_image(self):
        """Test image loading functionality."""
        with patch('src.depth_estimator.torch.hub.load') as mock_hub_load:
            # Mock the model and transforms
            mock_model = MagicMock()
            mock_transforms = MagicMock()
            mock_transform = MagicMock()
            
            mock_hub_load.side_effect = [mock_model, mock_transforms]
            mock_transforms.small_transform = mock_transform
            
            # Mock tensor creation
            mock_tensor = torch.randn(3, 256, 256)
            mock_transform.return_value = mock_tensor
            
            estimator = DepthEstimator(model_name="small")
            
            # Test image loading
            img_rgb, input_tensor = estimator.load_image(self.sample_image_path)
            
            # Verify results
            self.assertIsInstance(img_rgb, np.ndarray)
            self.assertEqual(img_rgb.shape, (240, 320, 3))
            self.assertEqual(input_tensor, mock_tensor)
    
    def test_load_nonexistent_image(self):
        """Test loading non-existent image."""
        with patch('src.depth_estimator.torch.hub.load') as mock_hub_load:
            mock_model = MagicMock()
            mock_transforms = MagicMock()
            mock_hub_load.side_effect = [mock_model, mock_transforms]
            
            estimator = DepthEstimator(model_name="small")
            
            with self.assertRaises(FileNotFoundError):
                estimator.load_image("nonexistent.jpg")
    
    def test_normalize_depth(self):
        """Test depth normalization."""
        with patch('src.depth_estimator.torch.hub.load') as mock_hub_load:
            mock_model = MagicMock()
            mock_transforms = MagicMock()
            mock_hub_load.side_effect = [mock_model, mock_transforms]
            
            estimator = DepthEstimator(model_name="small")
            
            # Test normal case
            depth_map = np.array([1, 2, 3, 4, 5])
            normalized = estimator._normalize_depth(depth_map)
            
            self.assertEqual(normalized.min(), 0.0)
            self.assertEqual(normalized.max(), 1.0)
            
            # Test edge case (all same values)
            depth_map_same = np.array([5, 5, 5, 5])
            normalized_same = estimator._normalize_depth(depth_map_same)
            
            self.assertTrue(np.allclose(normalized_same, 0.0))
    
    def test_get_depth_statistics(self):
        """Test depth statistics calculation."""
        with patch('src.depth_estimator.torch.hub.load') as mock_hub_load:
            mock_model = MagicMock()
            mock_transforms = MagicMock()
            mock_hub_load.side_effect = [mock_model, mock_transforms]
            
            estimator = DepthEstimator(model_name="small")
            
            # Create test depth map
            depth_map = np.random.rand(100, 100)
            stats = estimator.get_depth_statistics(depth_map)
            
            # Verify statistics keys
            expected_keys = [
                "mean_depth", "std_depth", "min_depth", 
                "max_depth", "median_depth", "depth_range"
            ]
            
            for key in expected_keys:
                self.assertIn(key, stats)
                self.assertIsInstance(stats[key], float)
    
    def test_save_depth_map(self):
        """Test depth map saving."""
        with patch('src.depth_estimator.torch.hub.load') as mock_hub_load:
            mock_model = MagicMock()
            mock_transforms = MagicMock()
            mock_hub_load.side_effect = [mock_model, mock_transforms]
            
            estimator = DepthEstimator(model_name="small")
            
            # Create test depth map
            depth_map = np.random.rand(100, 100)
            
            # Test PNG saving
            png_path = self.temp_dir / "test_depth.png"
            estimator.save_depth_map(depth_map, png_path, format="png")
            self.assertTrue(png_path.exists())
            
            # Test NPY saving
            npy_path = self.temp_dir / "test_depth.npy"
            estimator.save_depth_map(depth_map, npy_path, format="npy")
            self.assertTrue(npy_path.exists())
            
            # Verify NPY content
            loaded_depth = np.load(npy_path)
            np.testing.assert_array_equal(depth_map, loaded_depth)


class TestSampleImageCreation(unittest.TestCase):
    """Test cases for sample image creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_gradient_image(self):
        """Test gradient image creation."""
        output_path = self.temp_dir / "gradient.jpg"
        create_sample_image(output_path, pattern="gradient")
        
        self.assertTrue(output_path.exists())
        
        # Verify image properties
        img = cv2.imread(str(output_path))
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[:2], (480, 640))  # height, width
    
    def test_create_circles_image(self):
        """Test circles image creation."""
        output_path = self.temp_dir / "circles.jpg"
        create_sample_image(output_path, pattern="circles")
        
        self.assertTrue(output_path.exists())
        
        # Verify image properties
        img = cv2.imread(str(output_path))
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[:2], (480, 640))
    
    def test_create_checkerboard_image(self):
        """Test checkerboard image creation."""
        output_path = self.temp_dir / "checkerboard.jpg"
        create_sample_image(output_path, pattern="checkerboard")
        
        self.assertTrue(output_path.exists())
        
        # Verify image properties
        img = cv2.imread(str(output_path))
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[:2], (480, 640))
    
    def test_custom_size(self):
        """Test custom image size."""
        output_path = self.temp_dir / "custom.jpg"
        custom_size = (800, 600)
        create_sample_image(output_path, size=custom_size)
        
        self.assertTrue(output_path.exists())
        
        # Verify custom size
        img = cv2.imread(str(output_path))
        self.assertEqual(img.shape[:2], (600, 800))  # height, width


class TestConfigManager(unittest.TestCase):
    """Test cases for configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.yaml"
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        manager = ConfigManager(self.config_path)
        config = manager.get_config()
        
        # Verify default values
        self.assertEqual(config.model.name, "small")
        self.assertIsNone(config.model.device)
        self.assertEqual(config.visualization.colormap, "inferno")
        self.assertEqual(config.data.input_dir, "data/input")
    
    def test_config_save_and_load(self):
        """Test configuration saving and loading."""
        # Create and save config
        manager1 = ConfigManager(self.config_path)
        manager1.save_config()
        
        # Load config
        manager2 = ConfigManager(self.config_path)
        config = manager2.get_config()
        
        # Verify loaded values
        self.assertEqual(config.model.name, "small")
        self.assertEqual(config.visualization.colormap, "inferno")
    
    def test_config_update(self):
        """Test configuration updates."""
        manager = ConfigManager(self.config_path)
        
        # Update configuration
        manager.update_config(
            model={"name": "large"},
            visualization={"colormap": "viridis"}
        )
        
        config = manager.get_config()
        
        # Verify updates
        self.assertEqual(config.model.name, "large")
        self.assertEqual(config.visualization.colormap, "viridis")
    
    def test_invalid_config_key(self):
        """Test handling of invalid configuration keys."""
        manager = ConfigManager(self.config_path)
        
        # This should not raise an exception, just log a warning
        manager.update_config(invalid_key="value")
        
        # Configuration should remain unchanged
        config = manager.get_config()
        self.assertEqual(config.model.name, "small")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_image_path = self.temp_dir / "integration_test.jpg"
        
        # Create a test image
        create_sample_image(self.sample_image_path, size=(320, 240))
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.depth_estimator.torch.hub.load')
    def test_complete_pipeline(self, mock_hub_load):
        """Test complete depth estimation pipeline."""
        # Mock the model and transforms
        mock_model = MagicMock()
        mock_transforms = MagicMock()
        mock_transform = MagicMock()
        
        # Mock model prediction
        mock_prediction = torch.randn(1, 1, 240, 320)
        mock_model.return_value = mock_prediction
        
        mock_hub_load.side_effect = [mock_model, mock_transforms]
        mock_transforms.small_transform = mock_transform
        
        # Mock tensor creation
        mock_tensor = torch.randn(3, 256, 256)
        mock_transform.return_value = mock_tensor
        
        # Initialize estimator
        estimator = DepthEstimator(model_name="small")
        
        # Process image
        original_img, depth_map = estimator.process_image(self.sample_image_path)
        
        # Verify results
        self.assertIsInstance(original_img, np.ndarray)
        self.assertIsInstance(depth_map, np.ndarray)
        self.assertEqual(original_img.shape, (240, 320, 3))
        self.assertEqual(depth_map.shape, (240, 320))
        
        # Verify depth map is normalized
        self.assertGreaterEqual(depth_map.min(), 0.0)
        self.assertLessEqual(depth_map.max(), 1.0)
        
        # Test statistics
        stats = estimator.get_depth_statistics(depth_map)
        self.assertIn("mean_depth", stats)
        self.assertIn("max_depth", stats)
        
        # Test saving
        depth_save_path = self.temp_dir / "saved_depth.png"
        estimator.save_depth_map(depth_map, depth_save_path)
        self.assertTrue(depth_save_path.exists())


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDepthEstimator,
        TestSampleImageCreation,
        TestConfigManager,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
