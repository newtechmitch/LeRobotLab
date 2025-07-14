# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project requires the LeRobot library to be installed in a sibling directory. Setup using conda:

```bash
conda create -n lerobotlab python=3.10
conda activate lerobotlab
conda env update --file environment.yml
```

For minimal dataset exploration only:
```bash
conda install jupyter pandas requests tqdm ipywidgets -c conda-forge
pip install huggingface_hub
```

## Code Style and Linting

- Line length: 120 characters (pyproject.toml) / 140 characters (flake8)
- Use black, flake8, pylint, and ruff for code formatting and linting
- All tools configured for 120-character line length except flake8 (140 chars)

## Project Structure

### Core Components
- `notebooks/`: Jupyter notebooks for dataset exploration and analysis
  - `LeRobot_dataset_exploration.ipynb`: Comprehensive exploration of all HuggingFace lerobot datasets
  - `so100_so101_exploration.ipynb`: Focused analysis of SO100/SO101 robot datasets
- `scripts/`: Hardware interaction examples using LeRobot library
  - `camera.py`: OpenCV camera configuration and frame capture
  - `teleoperate.py`: SO101 robot teleoperation setup
- `configs/`: Configuration files (directory exists but empty)

### Generated Outputs
- CSV files with dataset metadata and analysis results
- `lerobot_datasets_videos.html`: Interactive web interface for dataset exploration
- `videos/`: Downloaded sample videos organized by dataset and camera

## Hardware Integration

The project includes examples for:
- OpenCV camera integration with LeRobot camera abstractions
- SO101 robot arm teleoperation using leader/follower configuration
- Port configuration for specific hardware devices (USB serial connections)

## Dependencies

Key dependencies include:
- `lerobot`: Core robotics library (installed from sibling directory)
- `huggingface_hub`: Dataset access and metadata retrieval
- `torch`, `torchvision`: Machine learning frameworks
- `opencv-python-headless`: Computer vision
- `pandas`, `numpy`: Data analysis
- `jupyter`: Interactive development