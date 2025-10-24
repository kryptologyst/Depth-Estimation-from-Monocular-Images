"""
Command-line interface for depth estimation.

This module provides a comprehensive CLI for batch processing images
and various depth estimation operations.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import json

from src.depth_estimator import DepthEstimator, create_sample_image
from src.config import ConfigManager, create_default_config_file


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def process_single_image(
    image_path: Path,
    estimator: DepthEstimator,
    output_dir: Path,
    save_visualization: bool = True,
    save_depth_map: bool = True,
    save_statistics: bool = True
) -> dict:
    """
    Process a single image and save results.
    
    Args:
        image_path: Path to input image
        estimator: Depth estimator instance
        output_dir: Output directory
        save_visualization: Whether to save visualization
        save_statistics: Whether to save statistics
        
    Returns:
        Dictionary with processing results
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Process image
        logger.info(f"Processing: {image_path}")
        original_img, depth_map = estimator.process_image(image_path)
        
        # Create output filename base
        output_base = output_dir / image_path.stem
        
        results = {
            "input_file": str(image_path),
            "status": "success",
            "output_files": []
        }
        
        # Save depth map
        if save_depth_map:
            depth_path = f"{output_base}_depth.png"
            estimator.save_depth_map(depth_map, depth_path)
            results["output_files"].append(depth_path)
            logger.info(f"Depth map saved: {depth_path}")
        
        # Save visualization
        if save_visualization:
            viz_path = f"{output_base}_visualization.png"
            estimator.visualize_results(
                original_img, 
                depth_map, 
                save_path=viz_path
            )
            results["output_files"].append(viz_path)
            logger.info(f"Visualization saved: {viz_path}")
        
        # Save statistics
        if save_statistics:
            stats = estimator.get_depth_statistics(depth_map)
            stats_path = f"{output_base}_statistics.json"
            
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            results["output_files"].append(stats_path)
            results["statistics"] = stats
            logger.info(f"Statistics saved: {stats_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to process {image_path}: {e}")
        return {
            "input_file": str(image_path),
            "status": "error",
            "error": str(e),
            "output_files": []
        }


def process_batch(
    input_paths: List[Path],
    estimator: DepthEstimator,
    output_dir: Path,
    **kwargs
) -> List[dict]:
    """
    Process multiple images in batch.
    
    Args:
        input_paths: List of input image paths
        estimator: Depth estimator instance
        output_dir: Output directory
        **kwargs: Additional arguments for process_single_image
        
    Returns:
        List of processing results
    """
    logger = logging.getLogger(__name__)
    results = []
    
    logger.info(f"Processing {len(input_paths)} images...")
    
    for i, image_path in enumerate(input_paths, 1):
        logger.info(f"Progress: {i}/{len(input_paths)}")
        
        result = process_single_image(
            image_path, 
            estimator, 
            output_dir, 
            **kwargs
        )
        results.append(result)
    
    return results


def find_images(input_path: Path, extensions: List[str] = None) -> List[Path]:
    """
    Find all image files in a directory.
    
    Args:
        input_path: Input path (file or directory)
        extensions: List of file extensions to search for
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    if input_path.is_file():
        if input_path.suffix.lower() in extensions:
            return [input_path]
        else:
            return []
    
    elif input_path.is_dir():
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    else:
        raise FileNotFoundError(f"Path not found: {input_path}")


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Depth estimation from monocular images using MiDaS models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python -m src.cli process image.jpg --output results/
  
  # Process all images in a directory
  python -m src.cli batch input_dir/ --output results/
  
  # Use a specific model
  python -m src.cli process image.jpg --model large --output results/
  
  # Generate sample images
  python -m src.cli sample --count 5 --output samples/
  
  # Create configuration file
  python -m src.cli config --create
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a single image')
    process_parser.add_argument('input', type=Path, help='Input image path')
    process_parser.add_argument('--output', '-o', type=Path, required=True, 
                               help='Output directory')
    process_parser.add_argument('--model', '-m', default='small',
                               choices=['small', 'medium', 'large', 'hybrid', 'large_hybrid'],
                               help='Model to use')
    process_parser.add_argument('--device', default=None,
                               help='Device to use (cpu, cuda, or auto)')
    process_parser.add_argument('--no-viz', action='store_true',
                               help='Skip visualization')
    process_parser.add_argument('--no-depth', action='store_true',
                               help='Skip depth map saving')
    process_parser.add_argument('--no-stats', action='store_true',
                               help='Skip statistics saving')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple images')
    batch_parser.add_argument('input', type=Path, help='Input directory or file')
    batch_parser.add_argument('--output', '-o', type=Path, required=True,
                             help='Output directory')
    batch_parser.add_argument('--model', '-m', default='small',
                             choices=['small', 'medium', 'large', 'hybrid', 'large_hybrid'],
                             help='Model to use')
    batch_parser.add_argument('--device', default=None,
                             help='Device to use (cpu, cuda, or auto)')
    batch_parser.add_argument('--extensions', nargs='+', 
                             default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                             help='File extensions to process')
    batch_parser.add_argument('--no-viz', action='store_true',
                             help='Skip visualization')
    batch_parser.add_argument('--no-depth', action='store_true',
                             help='Skip depth map saving')
    batch_parser.add_argument('--no-stats', action='store_true',
                             help='Skip statistics saving')
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Generate sample images')
    sample_parser.add_argument('--output', '-o', type=Path, required=True,
                              help='Output directory')
    sample_parser.add_argument('--count', '-c', type=int, default=3,
                              help='Number of sample images to generate')
    sample_parser.add_argument('--pattern', '-p', default='gradient',
                              choices=['gradient', 'circles', 'checkerboard'],
                              help='Pattern type')
    sample_parser.add_argument('--size', nargs=2, type=int, default=[640, 480],
                              help='Image size (width height)')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--create', action='store_true',
                              help='Create default configuration file')
    config_parser.add_argument('--path', type=Path, default='config/config.yaml',
                              help='Configuration file path')
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--config', type=Path,
                       help='Configuration file path')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == 'config':
            if args.create:
                create_default_config_file(args.path)
                logger.info(f"Configuration file created: {args.path}")
            else:
                logger.info("Use --create to generate a configuration file")
        
        elif args.command == 'sample':
            # Generate sample images
            args.output.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Generating {args.count} sample images...")
            
            for i in range(args.count):
                sample_path = args.output / f"sample_{i+1}_{args.pattern}.jpg"
                create_sample_image(
                    sample_path,
                    size=tuple(args.size),
                    pattern=args.pattern
                )
                logger.info(f"Created: {sample_path}")
        
        else:
            # Process images
            if args.command == 'process':
                input_paths = [args.input]
            else:  # batch
                input_paths = find_images(args.input, args.extensions)
                
                if not input_paths:
                    logger.error(f"No images found in {args.input}")
                    return
            
            # Create output directory
            args.output.mkdir(parents=True, exist_ok=True)
            
            # Initialize estimator
            logger.info(f"Initializing {args.model} model...")
            estimator = DepthEstimator(
                model_name=args.model,
                device=args.device
            )
            
            # Process images
            if args.command == 'process':
                results = [process_single_image(
                    args.input,
                    estimator,
                    args.output,
                    save_visualization=not args.no_viz,
                    save_depth_map=not args.no_depth,
                    save_statistics=not args.no_stats
                )]
            else:  # batch
                results = process_batch(
                    input_paths,
                    estimator,
                    args.output,
                    save_visualization=not args.no_viz,
                    save_depth_map=not args.no_depth,
                    save_statistics=not args.no_stats
                )
            
            # Save batch results summary
            if len(results) > 1:
                summary_path = args.output / "batch_results.json"
                with open(summary_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Batch results saved: {summary_path}")
            
            # Print summary
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = len(results) - successful
            
            logger.info(f"Processing complete: {successful} successful, {failed} failed")
            
            if failed > 0:
                logger.warning("Some images failed to process. Check logs for details.")
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
