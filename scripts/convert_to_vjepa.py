#!/usr/bin/env python3
"""
Convert LeRobot dataset to vjepa-format structure.

vjepa-format structure:
Dataset_folder/
├── dataset_list.txt          # CSV file with episode paths (one per line)
└── episodes/
    ├── episode_001/
    ├── episode_002/
    ├── ...
    └── episode_N/

episode_n/
├── metadata.json             # Episode metadata with file paths
├── trajectory.h5            # Robot trajectory and sensor data
└── recordings/
    └── MP4/
        └── front_camera.mp4     # front camera
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import pandas as pd
import numpy as np

# Configuration
CSV_PATH = "/Users/michelmeyer/Library/CloudStorage/Dropbox/Dev/LeRobotLab/notebooks/test-datasets.csv"
LEROBOT_DIR = "/Users/michelmeyer/Dropbox/Dev/LeRobotLab/downloaded_datasets"
OUTPUT_DATASET = "consolidated_vjepa_dataset"
VIDEO_KEY_COLUMN = "video_key"
DATASET_COLUMN = "dataset"

def load_dataset_info(dataset_path: Path) -> Dict:
    """Load dataset info from meta/info.json"""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info file not found: {info_path}")

    with open(info_path, 'r') as f:
        return json.load(f)


def load_episode_data(dataset_path: Path, episode_idx: int) -> pd.DataFrame:
    """Load episode data from parquet file"""
    episode_file = dataset_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    if not episode_file.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_file}")

    return pd.read_parquet(episode_file)


def create_trajectory_h5(episode_data: pd.DataFrame, output_path: Path):
    """Convert episode data to HDF5 trajectory format"""
    with h5py.File(output_path, 'w') as f:
        # Create groups for different data types
        action_group = f.create_group('action')
        observation_group = f.create_group('observation')
        metadata_group = f.create_group('metadata')

        # Save action data
        if 'action' in episode_data.columns:
            action_data = np.stack(episode_data['action'].values)
            action_group.create_dataset('data', data=action_data)
            
            # Add action names if available
            action_names = [
                "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
            ]
            action_group.attrs['names'] = action_names

        # Save observation state data
        if 'observation.state' in episode_data.columns:
            obs_data = np.stack(episode_data['observation.state'].values)
            observation_group.create_dataset('state', data=obs_data)
            
            # Add state names
            state_names = [
                "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
            ]
            observation_group.attrs['state_names'] = state_names

        # Save timestamps
        if 'timestamp' in episode_data.columns:
            timestamps = episode_data['timestamp'].values
            metadata_group.create_dataset('timestamp', data=timestamps)

        # Save frame indices
        if 'frame_index' in episode_data.columns:
            frame_indices = episode_data['frame_index'].values
            metadata_group.create_dataset('frame_index', data=frame_indices)

        # Add general metadata
        metadata_group.attrs['total_frames'] = len(episode_data)
        metadata_group.attrs['episode_length_s'] = len(episode_data) / 30.0  # Assuming 30fps
        metadata_group.attrs['fps'] = 30


def create_episode_metadata(episode_idx: int, episode_data: pd.DataFrame,
                          trajectory_path: Path, dataset_name: str) -> Dict:
    """Create metadata.json content for an episode"""
    metadata = {
        "episode_id": episode_idx,
        "episode_name": f"{dataset_name}-episode_{episode_idx:03d}",
        "source_dataset": dataset_name,
        "total_frames": len(episode_data),
        "duration_seconds": len(episode_data) / 30.0,  # Assuming 30fps
        "fps": 30,
        "files": {
            "trajectory": str(trajectory_path.name),
            "video": {
                "front_camera": str(Path("recordings/MP4/front_camera.mp4"))
            }
        },
        "data_keys": {
            "action": {
                "shape": [6],
                "dtype": "float32",
                "names": [
                    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
                ]
            },
            "observation.state": {
                "shape": [6],
                "dtype": "float32", 
                "names": [
                    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
                ]
            }
        },
        "task": "Grab orange ball"  # Based on the dataset
    }
    
    return metadata


def convert_datasets_to_vjepa(input_datasets: list, output_dataset_path: str, video_key: str = None):
    """Convert multiple LeRobot datasets to consolidated vjepa-format"""
    output_path = Path(output_dataset_path)
    
    print(f"Converting {len(input_datasets)} datasets to consolidated vjepa-format at {output_path}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_path / "episodes"
    episodes_dir.mkdir(exist_ok=True)
    
    # List to store episode paths for dataset_list.txt
    episode_paths = []
    total_episode_count = 0
    
    # Process each input dataset
    for dataset_config in input_datasets:
        dataset_name = dataset_config["name"]
        input_dataset_path = dataset_config["path"]
        # Use dataset-specific video key if provided
        current_video_key = dataset_config.get("video_key", video_key)
        if not current_video_key:
            print(f"Warning: No video key specified for dataset {dataset_name}. Skipping.")
            continue
            
        input_path = Path(input_dataset_path)
        
        if not input_path.exists():
            print(f"Warning: Input dataset not found: {input_path}")
            continue
            
        print(f"\nProcessing dataset '{dataset_name}' from {input_path}")
        
        # Load dataset info
        try:
            dataset_info = load_dataset_info(input_path)
        except Exception as e:
            print(f"Error loading dataset info for {dataset_name}: {e}")
            continue
        
        # Process each episode in this dataset
        videos_dir = input_path / "videos" / "chunk-000" / current_video_key
        dataset_episode_count = 0
        
        for episode_file in sorted((input_path / "data" / "chunk-000").glob("episode_*.parquet")):
            episode_idx = int(episode_file.stem.split('_')[1])
            dataset_episode_count += 1
            total_episode_count += 1
            
            print(f"  Processing episode {episode_idx}...")
            
            try:
                # Load episode data
                episode_data = load_episode_data(input_path, episode_idx)
                
                # Find corresponding video file
                video_path = videos_dir / f"episode_{episode_idx:06d}.mp4"
                
                if not video_path.exists():
                    print(f"    Warning: Video file not found: {video_path}")
                    continue
                
                # Create episode directory with dataset name prefix
                episode_dir_name = f"{dataset_name}-episode_{dataset_episode_count:03d}"
                episode_dir = episodes_dir / episode_dir_name
                episode_dir.mkdir(exist_ok=True)
                
                # Create recordings/MP4 directory
                recordings_dir = episode_dir / "recordings" / "MP4"
                recordings_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy video file
                output_video_path = recordings_dir / "front_camera.mp4"
                shutil.copy2(video_path, output_video_path)
                
                # Create trajectory.h5
                trajectory_path = episode_dir / "trajectory.h5"
                create_trajectory_h5(episode_data, trajectory_path)
                
                # Create metadata.json
                metadata = create_episode_metadata(dataset_episode_count, episode_data,
                                                 trajectory_path, dataset_name)
                metadata_path = episode_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Add to episode paths list
                episode_paths.append(f"episodes/{episode_dir_name}")
                
                print(f"    Created {episode_dir_name} with {len(episode_data)} frames")
                
            except Exception as e:
                print(f"    Error processing episode {episode_idx}: {e}")
                continue
        
        print(f"  Completed dataset '{dataset_name}': {dataset_episode_count} episodes")
    
    # Create dataset_list.txt - single file with all episodes
    dataset_list_path = output_path / "dataset_list.txt"
    with open(dataset_list_path, 'w') as f:
        for episode_path in episode_paths:
            f.write(f"{episode_path}\n")
    
    print(f"\nConsolidation complete!")
    print(f"Total episodes created: {total_episode_count}")
    print(f"Output directory: {output_path}")
    print(f"Dataset list saved to: {dataset_list_path}")


def load_datasets_from_csv(csv_path, lerobot_dir=LEROBOT_DIR, 
                         dataset_column=DATASET_COLUMN, 
                         video_key_column=VIDEO_KEY_COLUMN):
    """
    Load datasets information from a CSV file
    
    Args:
        csv_path: Path to the CSV file containing dataset info
        lerobot_dir: Base directory for downloaded datasets
        dataset_column: Name of column containing datasets in format "username/foldername"
        video_key_column: Name of column containing video key
        
    Returns:
        list: List of dataset configs and dict of video keys
    """
    try:
        # Load CSV file
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Verify required columns exist
        if dataset_column not in df.columns:
            raise ValueError(f"Dataset column '{dataset_column}' not found in CSV")
        if video_key_column not in df.columns:
            raise ValueError(f"Video key column '{video_key_column}' not found in CSV")
            
        # Build dataset configs
        dataset_configs = []
        video_keys = {}
        
        for _, row in df.iterrows():
            dataset = row[dataset_column]
            video_key = row[video_key_column]
            
            # Parse username and foldername from dataset
            username, foldername = dataset.split('/')
            
            # Build dataset path
            dataset_path = Path(lerobot_dir) / username / foldername
            
            # Use full dataset name with '+' instead of '/' for folder naming
            display_name = dataset.replace('/', '+')
            
            dataset_config = {
                "name": display_name,  # Use username+foldername for episode naming
                "path": str(dataset_path),
                "username": username,
                "foldername": foldername,  # Store original foldername separately 
                "full_name": dataset
            }
            
            dataset_configs.append(dataset_config)
            video_keys[display_name] = f"observation.images.{video_key}"
            
        return dataset_configs, video_keys
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return [], {}


def main():
    """Main function"""
    # Use configuration directly instead of parsing arguments
    print(f"Loading datasets from CSV: {CSV_PATH}")
    dataset_configs, video_keys = load_datasets_from_csv(CSV_PATH, LEROBOT_DIR, DATASET_COLUMN, VIDEO_KEY_COLUMN)
    
    if not dataset_configs:
        print("No datasets loaded from CSV. Exiting.")
        return
        
    print(f"Found {len(dataset_configs)} datasets in CSV")
    
    # Create consolidated output directory
    output_path = Path(OUTPUT_DATASET)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all datasets into a single consolidated output
    all_dataset_configs = []
    
    for dataset_config in dataset_configs:
        dataset_name = dataset_config["name"]
        video_key = video_keys.get(dataset_name, f"observation.images.{VIDEO_KEY_COLUMN}")
        
        print(f"Preparing dataset: {dataset_name}")
        print(f"  Path: {dataset_config['path']}")
        print(f"  Video key: {video_key}")
        
        # Store the video key in the dataset config for later use
        dataset_config["video_key"] = video_key
        all_dataset_configs.append(dataset_config)
    
    # Convert all datasets at once to a single consolidated directory
    print(f"\nConverting all datasets to consolidated directory: {output_path}")
    convert_datasets_to_vjepa(all_dataset_configs, str(output_path))
    
    print(f"All datasets consolidated into: {OUTPUT_DATASET}")


if __name__ == "__main__":
    main()