# LeRobotLab

Personal exploration and experimentation with LeRobot datasets and models.

## Project Overview

This repository contains experiments and analysis tools for working with LeRobot datasets from the HuggingFace Hub, with a focus on SO100 and SO101 robot datasets.

### Quick Setup for Dataset Exploration

For just exploring datasets (this project's focus), you can use a minimal setup d

```bash
# Create conda environment
conda create -n lerobotlab python=3.10
conda activate lerobotlab

# Install dependencies
conda install jupyter pandas requests tqdm ipywidgets -c conda-forge
pip install huggingface_hub
```


The project folder should be next to a LeRobot install. 
For the  LeRobot installation, follow the official installation guide:
[LeRobot Installation Instructions](https://github.com/huggingface/lerobot#installation)

## Contents

### SO100-SO101 Exploration Notebook

**`so100_so101_exploration.ipynb`**: Comprehensive search and analysis of SO100/SO101 robot datasets
- Searches HuggingFace Hub for relevant datasets
- Extracts metadata, licensing, and size information
- Generates clean CSV output for analysis
- Handles datasets with proper size estimation and error handling

#### Generated Output
- `so100_so101_datasets_analysis.csv`: Clean dataset analysis output with metadata and online links to view each dataset including videos.
- `so100_so101_all_datasets.csv`: Complete raw dataset information from HuggingFace Hub
- `so100_so101_license_summary.csv`: Summary of dataset licensing information 

### LeRobot Dataset Exploration Notebook

**`LeRobot_dataset_exploration.ipynb`**: Comprehensive exploration of all HuggingFace datasets authored by "lerobot"
- Discovers and processes metadata from all datasets under the `lerobot/` namespace on HuggingFace Hub
- Covers a wide variety of robot hardware: ALOHA, Franka, UR5, Sawyer, Stretch, and many more
- Extracts detailed information: episodes, samples, FPS, robot types, camera configurations
- Downloads sample videos from datasets with camera data for local preview
- Generates interactive HTML visualization with embedded video players
- Creates organized folder structure for offline browsing
- Provides parallel processing for efficient dataset handling

#### Generated Output
- `lerobot_datasets_videos.html`: Interactive web interface for dataset exploration
- `videos/`: Local video samples organized by dataset and camera

## About LeRobot

🤗 LeRobot is an open-source project providing models, datasets, and tools for real-world robotics in PyTorch. Learn more at [huggingface.co/lerobot](https://huggingface.co/lerobot).

This project focuses specifically on exploring and analyzing the SO100/SO101 datasets available through the LeRobot ecosystem.
