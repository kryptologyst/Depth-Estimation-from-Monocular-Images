# Depth Estimation from Monocular Images

A comprehensive implementation of depth estimation from single images using Intel's MiDaS models. This project provides multiple interfaces (CLI, Web UI, Python API) with enhanced visualization, configuration management, and robust error handling.

## Features

- **Multiple MiDaS Models**: Support for small, medium, large, and hybrid DPT models
- **Modern Architecture**: Clean, modular codebase with type hints and comprehensive documentation
- **Multiple Interfaces**: 
  - **Streamlit Web App**: Interactive web interface with real-time visualization
  - **Command Line Interface**: Batch processing and automation
  - **Python API**: Easy integration into your projects
- **Enhanced Visualization**: Multiple colormaps, 3D surface plots, overlays, and statistics
- **Configuration Management**: YAML-based configuration with sensible defaults
- **Comprehensive Testing**: Unit tests and integration tests
- **Error Handling**: Robust error handling with detailed logging
- **Sample Data**: Built-in sample image generation for testing

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Depth-Estimation-from-Monocular-Images.git
   cd Depth-Estimation-from-Monocular-Images
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web interface**:
   ```bash
   streamlit run web_app/app.py
   ```

4. **Or use the CLI**:
   ```bash
   python -m src.cli sample --output data/samples --count 3
   python -m src.cli process data/samples/sample_1_gradient.jpg --output results/
   ```

## Usage Examples

### Web Interface

Launch the Streamlit app for an interactive experience:

```bash
streamlit run web_app/app.py
```

Features:
- Upload images or generate samples
- Real-time depth estimation
- Multiple visualization modes
- Download results in various formats
- Model selection and configuration

### Command Line Interface

#### Process a single image:
```bash
python -m src.cli process image.jpg --output results/ --model large
```

#### Batch process multiple images:
```bash
python -m src.cli batch input_folder/ --output results/ --model medium
```

#### Generate sample images:
```bash
python -m src.cli sample --output samples/ --count 5 --pattern circles
```

#### Create configuration file:
```bash
python -m src.cli config --create
```

### Python API

```python
from src.depth_estimator import DepthEstimator
from src.config import load_config

# Load configuration
config = load_config()

# Initialize estimator
estimator = DepthEstimator(
    model_name="small",
    device="cuda"  # or "cpu"
)

# Process image
original_img, depth_map = estimator.process_image("path/to/image.jpg")

# Visualize results
estimator.visualize_results(original_img, depth_map)

# Get statistics
stats = estimator.get_depth_statistics(depth_map)
print(f"Mean depth: {stats['mean_depth']:.4f}")

# Save depth map
estimator.save_depth_map(depth_map, "output/depth.png")
```

## Project Structure

```
depth-estimation-monocular-images/
├── src/                          # Source code
│   ├── __init__.py
│   ├── depth_estimator.py       # Core depth estimation logic
│   ├── config.py                 # Configuration management
│   └── cli.py                    # Command-line interface
├── web_app/                      # Web interface
│   └── app.py                    # Streamlit application
├── tests/                        # Test suite
│   └── test_depth_estimation.py   # Unit and integration tests
├── config/                       # Configuration files
│   └── config.yaml               # Default configuration
├── data/                         # Data directories
│   ├── input/                    # Input images
│   └── output/                   # Output results
├── models/                       # Model storage
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Configuration

The application uses YAML configuration files for flexible setup. Create a configuration file:

```bash
python -m src.cli config --create
```

Example configuration (`config/config.yaml`):

```yaml
model:
  name: small                    # Model to use: small, medium, large, hybrid, large_hybrid
  device: null                   # Device: null (auto), cpu, cuda
  force_reload: false           # Force model reload

visualization:
  figsize: [15, 6]              # Figure size for plots
  colormap: inferno              # Colormap for depth visualization
  overlay_alpha: 0.6             # Transparency for overlays
  dpi: 300                       # DPI for saved images
  save_format: png               # Default save format

data:
  input_dir: data/input           # Default input directory
  output_dir: data/output         # Default output directory
  supported_formats:             # Supported image formats
    - .jpg
    - .jpeg
    - .png
    - .bmp
    - .tiff

logging:
  level: INFO                    # Logging level
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: null                # Log file path (null for console only)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_depth_estimation.py -v
```

## Available Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `small` | ~50MB | Fast | Good | Real-time, mobile |
| `medium` | ~100MB | Medium | Better | Balanced performance |
| `large` | ~200MB | Slow | Best | High accuracy needed |
| `hybrid` | ~150MB | Medium | Excellent | Modern architecture |
| `large_hybrid` | ~300MB | Slow | Excellent | Maximum accuracy |

## Visualization Features

- **Multiple Colormaps**: inferno, viridis, plasma, magma, jet, hot, cool
- **3D Surface Plots**: Interactive 3D visualization using Plotly
- **Depth Overlays**: Blend depth information with original images
- **Statistics Dashboard**: Comprehensive depth analysis
- **Export Options**: PNG, NPY, TIFF formats

## 🔧 Advanced Usage

### Custom Model Integration

```python
# Use custom transforms
estimator = DepthEstimator(model_name="large")
estimator.transform = custom_transform

# Process with custom preprocessing
img_rgb, input_tensor = estimator.load_image("image.jpg")
# Apply custom preprocessing to input_tensor
depth_map = estimator.estimate_depth(input_tensor, img_rgb.shape[:2])
```

### Batch Processing with Progress Tracking

```python
from pathlib import Path
from tqdm import tqdm

input_dir = Path("input_images")
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

estimator = DepthEstimator(model_name="medium")

image_files = list(input_dir.glob("*.jpg"))
for img_path in tqdm(image_files):
    try:
        original_img, depth_map = estimator.process_image(img_path)
        estimator.save_depth_map(depth_map, output_dir / f"{img_path.stem}_depth.png")
    except Exception as e:
        print(f"Failed to process {img_path}: {e}")
```

### Configuration Management

```python
from src.config import ConfigManager

# Load and modify configuration
config_manager = ConfigManager("custom_config.yaml")
config_manager.update_config(
    model={"name": "large"},
    visualization={"colormap": "viridis"}
)
config_manager.save_config()
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Use smaller model: `model_name="small"`
   - Process smaller images
   - Use CPU: `device="cpu"`

2. **Model Download Issues**:
   - Check internet connection
   - Use `force_reload=True` to retry download
   - Verify PyTorch installation

3. **Image Loading Errors**:
   - Check file format support
   - Verify file permissions
   - Ensure image is not corrupted

### Performance Optimization

- **GPU Acceleration**: Use CUDA-enabled PyTorch
- **Batch Processing**: Process multiple images together
- **Image Resizing**: Resize large images before processing
- **Model Caching**: Models are cached after first load

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `python -m pytest tests/`
6. Commit your changes: `git commit -am 'Add feature'`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Intel MiDaS Team**: For the excellent depth estimation models
- **PyTorch Team**: For the deep learning framework
- **Streamlit Team**: For the web interface framework
- **OpenCV Team**: For computer vision tools

## References

- [MiDaS Paper](https://arxiv.org/abs/1907.01341)
- [MiDaS GitHub](https://github.com/isl-org/MiDaS)
- [PyTorch Hub](https://pytorch.org/hub/)

## Related Projects

- [DPT (Dense Prediction Transformer)](https://github.com/isl-org/DPT)
- [MiDaS v3](https://github.com/isl-org/MiDaS/releases/tag/v3_1)
- [Depth Anything](https://github.com/DepthAnything/Depth-Anything)


# Depth-Estimation-from-Monocular-Images
