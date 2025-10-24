"""
Streamlit web interface for depth estimation.

This module provides a modern, interactive web interface for the depth
estimation application using Streamlit.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import tempfile
import logging
from typing import Optional, Tuple

# Import our modules
from src.depth_estimator import DepthEstimator, create_sample_image
from src.config import ConfigManager, create_default_config_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Depth Estimation from Monocular Images",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5a8a;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'estimator' not in st.session_state:
        st.session_state.estimator = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'depth_map' not in st.session_state:
        st.session_state.depth_map = None
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager()


def create_estimator(model_name: str, device: Optional[str] = None) -> DepthEstimator:
    """Create and cache depth estimator."""
    cache_key = f"estimator_{model_name}_{device}"
    
    if cache_key not in st.session_state:
        with st.spinner(f"Loading {model_name} model..."):
            try:
                st.session_state[cache_key] = DepthEstimator(
                    model_name=model_name,
                    device=device
                )
                st.success(f"✅ {model_name} model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load model: {e}")
                return None
    
    return st.session_state[cache_key]


def process_uploaded_image(
    uploaded_file, 
    estimator: DepthEstimator
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Process uploaded image and return original and depth map."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process image
        with st.spinner("Processing image..."):
            original_img, depth_map = estimator.process_image(tmp_path)
        
        # Clean up temporary file
        Path(tmp_path).unlink()
        
        return original_img, depth_map
        
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None, None


def create_depth_visualization(
    original_img: np.ndarray, 
    depth_map: np.ndarray,
    colormap: str = "inferno"
) -> None:
    """Create interactive depth visualization."""
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎨 Depth Map", "🔄 Overlay", "📈 3D Surface"])
    
    with tab1:
        # Overview with side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(original_img, use_column_width=True)
        
        with col2:
            st.subheader("Depth Map")
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(depth_map, cmap=colormap)
            ax.set_title("Depth Map")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
    
    with tab2:
        # Enhanced depth map visualization
        st.subheader("Enhanced Depth Visualization")
        
        # Colormap selection
        colormap_options = ["inferno", "viridis", "plasma", "magma", "jet", "hot", "cool"]
        selected_colormap = st.selectbox("Select Colormap", colormap_options, index=0)
        
        # Create enhanced visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original depth map
        im1 = ax1.imshow(depth_map, cmap=selected_colormap)
        ax1.set_title(f"Depth Map ({selected_colormap})")
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Histogram
        ax2.hist(depth_map.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title("Depth Distribution")
        ax2.set_xlabel("Depth Value")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with tab3:
        # Overlay visualization
        st.subheader("Depth Overlay")
        
        # Alpha slider
        alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.6, 0.1)
        
        # Create overlay
        estimator = st.session_state.estimator
        overlay = estimator._create_overlay(original_img, depth_map, alpha)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(overlay, caption=f"Overlay (α={alpha})", use_column_width=True)
        
        with col2:
            # Show depth statistics
            stats = estimator.get_depth_statistics(depth_map)
            
            st.subheader("Depth Statistics")
            for key, value in stats.items():
                st.metric(
                    key.replace("_", " ").title(),
                    f"{value:.4f}"
                )
    
    with tab4:
        # 3D surface plot
        st.subheader("3D Depth Surface")
        
        # Downsample for performance
        downsample_factor = st.slider("Downsample Factor", 1, 10, 4)
        
        if st.button("Generate 3D Surface"):
            with st.spinner("Generating 3D surface..."):
                # Downsample depth map
                h, w = depth_map.shape
                new_h, new_w = h // downsample_factor, w // downsample_factor
                
                depth_downsampled = cv2.resize(depth_map, (new_w, new_h))
                
                # Create 3D surface
                x = np.arange(new_w)
                y = np.arange(new_h)
                X, Y = np.meshgrid(x, y)
                
                fig = go.Figure(data=[go.Surface(
                    z=depth_downsampled,
                    x=X,
                    y=Y,
                    colorscale=colormap,
                    name='Depth Surface'
                )])
                
                fig.update_layout(
                    title="3D Depth Surface",
                    scene=dict(
                        xaxis_title="Width",
                        yaxis_title="Height",
                        zaxis_title="Depth"
                    ),
                    width=800,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Depth Estimation from Monocular Images</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application uses Intel's MiDaS models to estimate depth maps from single images.
    Upload an image below to see the depth estimation in action!
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "Small (Fast)": "small",
            "Medium (Balanced)": "medium", 
            "Large (Accurate)": "large",
            "Hybrid (DPT)": "hybrid",
            "Large Hybrid (DPT)": "large_hybrid"
        }
        
        selected_model = st.selectbox(
            "Select Model",
            list(model_options.keys()),
            index=0,
            help="Larger models are more accurate but slower"
        )
        
        model_name = model_options[selected_model]
        
        # Device selection
        device_options = ["Auto", "CPU", "GPU (CUDA)"]
        device_choice = st.selectbox("Device", device_options, index=0)
        
        device = None if device_choice == "Auto" else device_choice.lower()
        
        # Visualization options
        st.header("🎨 Visualization")
        colormap = st.selectbox(
            "Colormap",
            ["inferno", "viridis", "plasma", "magma", "jet", "hot", "cool"],
            index=0
        )
        
        # Create estimator
        if st.button("🔄 Initialize Model"):
            st.session_state.estimator = create_estimator(model_name, device)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Image")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to estimate its depth map"
        )
        
        # Sample image option
        if st.button("🎲 Generate Sample Image"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                sample_path = tmp_file.name
            
            pattern = st.selectbox("Sample Pattern", ["gradient", "circles", "checkerboard"])
            create_sample_image(sample_path, pattern=pattern)
            
            # Process sample image
            if st.session_state.estimator:
                original_img, depth_map = st.session_state.estimator.process_image(sample_path)
                st.session_state.processed_image = original_img
                st.session_state.depth_map = depth_map
                
                # Clean up
                Path(sample_path).unlink()
                
                st.success("✅ Sample image processed!")
            else:
                st.warning("⚠️ Please initialize a model first")
    
    with col2:
        st.header("ℹ️ Information")
        
        st.info("""
        **How it works:**
        1. Upload an image or generate a sample
        2. The MiDaS model estimates depth for each pixel
        3. Visualize results in multiple formats
        
        **Tips:**
        - Use images with clear depth cues
        - Larger models provide better accuracy
        - GPU acceleration speeds up processing
        """)
        
        if st.session_state.estimator:
            st.success("✅ Model ready!")
        else:
            st.warning("⚠️ Initialize model in sidebar")
    
    # Process uploaded image
    if uploaded_file is not None and st.session_state.estimator:
        original_img, depth_map = process_uploaded_image(uploaded_file, st.session_state.estimator)
        
        if original_img is not None and depth_map is not None:
            st.session_state.processed_image = original_img
            st.session_state.depth_map = depth_map
    
    # Display results
    if st.session_state.processed_image is not None and st.session_state.depth_map is not None:
        st.header("📊 Results")
        create_depth_visualization(
            st.session_state.processed_image, 
            st.session_state.depth_map,
            colormap
        )
        
        # Download options
        st.header("💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Depth Map"):
                # Convert depth map to downloadable format
                depth_scaled = (st.session_state.depth_map * 255).astype(np.uint8)
                depth_pil = Image.fromarray(depth_scaled)
                
                # Create download link
                import io
                buf = io.BytesIO()
                depth_pil.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download PNG",
                    data=byte_im,
                    file_name="depth_map.png",
                    mime="image/png"
                )
        
        with col2:
            if st.button("📊 Download Statistics"):
                stats = st.session_state.estimator.get_depth_statistics(st.session_state.depth_map)
                
                # Create CSV
                import pandas as pd
                df = pd.DataFrame([stats])
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="depth_statistics.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📈 Download Raw Data"):
                # Save as numpy array
                import io
                buf = io.BytesIO()
                np.save(buf, st.session_state.depth_map)
                byte_data = buf.getvalue()
                
                st.download_button(
                    label="Download NPY",
                    data=byte_data,
                    file_name="depth_map.npy",
                    mime="application/octet-stream"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit and Intel MiDaS models
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
