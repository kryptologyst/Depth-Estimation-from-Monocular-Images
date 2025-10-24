#!/usr/bin/env python3
"""
Setup script for depth estimation project.

This script handles the initial setup and installation of the project.
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Run a command and handle errors."""
    logger.info(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8 or higher is required")
        return False
    
    logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def install_dependencies():
    """Install project dependencies."""
    logger.info("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        logger.error("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )


def create_directories():
    """Create necessary directories."""
    logger.info("📁 Creating project directories...")
    
    directories = [
        "data/input",
        "data/output", 
        "models",
        "logs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"   ✅ Created: {directory}")
    
    return True


def create_config():
    """Create default configuration."""
    logger.info("⚙️ Creating configuration...")
    
    try:
        # Import and create config
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from src.config import create_default_config_file
        
        create_default_config_file("config/config.yaml")
        logger.info("   ✅ Configuration file created: config/config.yaml")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Failed to create configuration: {e}")
        return False


def create_sample_data():
    """Create sample data for testing."""
    logger.info("🎨 Creating sample data...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from src.depth_estimator import create_sample_image
        
        sample_images = [
            ("data/input/sample_gradient.jpg", "gradient"),
            ("data/input/sample_circles.jpg", "circles"),
            ("data/input/sample_checkerboard.jpg", "checkerboard")
        ]
        
        for path, pattern in sample_images:
            create_sample_image(path, pattern=pattern)
            logger.info(f"   ✅ Created: {path}")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Failed to create sample data: {e}")
        return False


def run_tests():
    """Run the test suite."""
    logger.info("🧪 Running tests...")
    
    return run_command(
        f"{sys.executable} -m pytest tests/ -v",
        "Running test suite"
    )


def main():
    """Main setup function."""
    print("🚀 Depth Estimation Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        logger.warning("⚠️ Dependency installation failed. You may need to install manually:")
        logger.warning("   pip install -r requirements.txt")
    
    # Create configuration
    if not create_config():
        logger.warning("⚠️ Configuration creation failed")
    
    # Create sample data
    if not create_sample_data():
        logger.warning("⚠️ Sample data creation failed")
    
    # Run tests (optional)
    test_choice = input("\n🧪 Run tests? (y/n): ").lower().strip()
    if test_choice in ['y', 'yes']:
        run_tests()
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("   1. 🌐 Run web interface: streamlit run web_app/app.py")
    print("   2. 💻 Try CLI: python -m src.cli --help")
    print("   3. 🎮 Run demo: python demo.py")
    print("   4. 📖 Read README.md for detailed usage")
    
    print("\n🔗 Quick commands:")
    print("   • Web app: streamlit run web_app/app.py")
    print("   • CLI help: python -m src.cli --help")
    print("   • Process image: python -m src.cli process data/input/sample_gradient.jpg --output results/")
    print("   • Generate samples: python -m src.cli sample --output samples/ --count 5")


if __name__ == "__main__":
    main()
