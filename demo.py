#!/usr/bin/env python3
"""
Demo script for depth estimation project.

This script demonstrates the core functionality of the depth estimation system
with sample images and various visualization options.
"""

import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.depth_estimator import DepthEstimator, create_sample_image
from src.config import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run the demo."""
    print("🔍 Depth Estimation Demo")
    print("=" * 50)
    
    # Initialize configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    print(f"📋 Configuration loaded:")
    print(f"   Model: {config.model.name}")
    print(f"   Device: {config.model.device or 'Auto'}")
    print(f"   Colormap: {config.visualization.colormap}")
    print()
    
    # Create sample images if they don't exist
    sample_dir = Path("data/input")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    sample_images = [
        ("gradient", "sample_gradient.jpg"),
        ("circles", "sample_circles.jpg"),
        ("checkerboard", "sample_checkerboard.jpg")
    ]
    
    print("🎨 Creating sample images...")
    for pattern, filename in sample_images:
        sample_path = sample_dir / filename
        if not sample_path.exists():
            create_sample_image(sample_path, pattern=pattern)
            print(f"   ✅ Created: {sample_path}")
        else:
            print(f"   ⏭️  Exists: {sample_path}")
    
    print()
    
    # Initialize depth estimator
    print("🤖 Initializing depth estimator...")
    try:
        estimator = DepthEstimator(
            model_name=config.model.name,
            device=config.model.device
        )
        print(f"   ✅ Model '{config.model.name}' loaded successfully")
        print(f"   📱 Device: {estimator.device}")
        print()
    except Exception as e:
        print(f"   ❌ Failed to initialize model: {e}")
        print("   💡 Try installing PyTorch: pip install torch torchvision")
        return
    
    # Process sample images
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Processing sample images...")
    for pattern, filename in sample_images:
        sample_path = sample_dir / filename
        
        try:
            print(f"   Processing: {sample_path.name}")
            
            # Process image
            original_img, depth_map = estimator.process_image(sample_path)
            
            # Save results
            output_base = output_dir / sample_path.stem
            
            # Save depth map
            depth_path = f"{output_base}_depth.png"
            estimator.save_depth_map(depth_map, depth_path)
            
            # Save visualization
            viz_path = f"{output_base}_visualization.png"
            estimator.visualize_results(
                original_img, 
                depth_map, 
                save_path=viz_path,
                colormap=config.visualization.colormap
            )
            
            # Get and display statistics
            stats = estimator.get_depth_statistics(depth_map)
            print(f"      📊 Mean depth: {stats['mean_depth']:.4f}")
            print(f"      📊 Depth range: {stats['depth_range']:.4f}")
            print(f"      💾 Saved: {depth_path}")
            print(f"      🖼️  Saved: {viz_path}")
            
        except Exception as e:
            print(f"      ❌ Error processing {sample_path.name}: {e}")
        
        print()
    
    print("🎉 Demo completed successfully!")
    print()
    print("📁 Output files saved to: data/output/")
    print("🌐 To run the web interface: streamlit run web_app/app.py")
    print("💻 To use CLI: python -m src.cli --help")
    print()
    print("🔗 Next steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Run web app: streamlit run web_app/app.py")
    print("   3. Try CLI: python -m src.cli process data/input/sample_gradient.jpg --output results/")


if __name__ == "__main__":
    main()
