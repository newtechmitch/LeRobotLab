# LeRobot Dataset Explorer

A comprehensive exploration system for LeRobot datasets from HuggingFace Hub that discovers, analyzes, and visualizes robot learning datasets with interactive video content.

## Overview

This project provides tools to:
- 🔍 **Discover** all available LeRobot datasets from HuggingFace Hub
- 📊 **Collect** detailed metadata from 100+ datasets
- 🎥 **Download** sample videos from datasets with camera data
- 🌐 **Generate** interactive HTML viewer for browsing datasets
- 📈 **Analyze** dataset statistics and categorization

## Project Structure

```
LeRobotLab/
├── notebooks/
│   ├── dataset_exploration_clean.ipynb    # Main exploration notebook
│   ├── lerobot_datasets_videos.html       # Generated HTML viewer
│   └── videos/                            # Downloaded dataset videos
├── configs/                               # Configuration files
├── data/                                  # Data storage
├── outputs/                               # Model outputs and logs
├── scripts/                               # Utility scripts
├── environment.yml                        # Conda environment
├── requirements.txt                       # Python dependencies
└── README.md                             # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate lerobot

# Or use pip
pip install -r requirements.txt
```

### 2. Run the Explorer

Open and run the notebook:
```bash
jupyter notebook notebooks/dataset_exploration_clean.ipynb
```

### 3. View Results

The notebook generates an interactive HTML viewer:
- Open `notebooks/lerobot_datasets_videos.html` in your browser
- Browse 90+ datasets with embedded videos
- Explore dataset statistics and categories

## Features

### Dataset Discovery
- Automatically discovers all LeRobot datasets from HuggingFace Hub
- Currently processes 100+ datasets across multiple robot platforms

### Metadata Collection
- Extracts detailed metadata: episodes, samples, FPS, robot types
- Identifies datasets with camera data for video processing
- Handles error cases gracefully with comprehensive logging

### Video Downloads
- Downloads sample videos from datasets with camera data
- Parallel processing for efficient downloading
- Organizes videos by dataset and camera view

### Interactive Visualization
- Beautiful HTML interface with modern CSS styling
- Embedded video players for direct viewing
- Dataset statistics and category breakdown
- Responsive design for different screen sizes

### Analytics
- Dataset family analysis (aloha, berkeley, xarm, etc.)
- Episode and sample count aggregation
- Camera availability tracking
- Download success/failure reporting

## Dataset Categories

The explorer currently processes datasets from these major categories:
- **aloha**: ALOHA robot manipulation tasks
- **berkeley**: Berkeley robot learning datasets
- **xarm**: xArm robot demonstrations
- **austin**: Austin robot datasets
- **columbia**: Columbia University datasets
- **utokyo**: University of Tokyo datasets
- And many more...

## Technical Details

### Dependencies
- `pandas`: Data manipulation and analysis
- `requests`: HTTP requests for video downloads
- `huggingface_hub`: HuggingFace API access
- `lerobot`: LeRobot dataset loading (external dependency)
- `concurrent.futures`: Parallel processing

### Performance
- Processes 100+ datasets in under 10 minutes
- Parallel video downloads (3 workers by default)
- Rate limiting to respect API limits
- Comprehensive error handling and retry logic

## Generated Output

The project generates:
1. **Comprehensive Dataset Metrics** - DataFrame with all dataset metadata
2. **Video Library** - Organized collection of sample videos
3. **Interactive HTML Viewer** - Beautiful web interface for exploration
4. **Analysis Reports** - Statistical summaries and breakdowns

## Configuration

Key configuration options in the notebook:
- `datasets_to_process`: Control which datasets to process
- `max_datasets`: Limit video downloads for testing
- `max_workers`: Parallel download workers
- `videos_folder`: Video storage location

## Contributing

This project is part of the LeRobot ecosystem. For contributions:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project follows the same license as the LeRobot project.

## Related Projects

- [LeRobot](https://github.com/huggingface/lerobot) - Main LeRobot framework
- [HuggingFace Hub](https://huggingface.co/lerobot) - Dataset hosting

## Acknowledgments

Built on top of the excellent LeRobot framework by HuggingFace. Special thanks to the robotics community for contributing datasets and tools.

---

**Generated**: May 2025 | **Status**: Active Development
