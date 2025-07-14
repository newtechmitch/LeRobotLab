#!/usr/bin/env python3
"""
Transform LeRobot dataset by selecting one video stream, cutting episodes into 3-second chunks at 4fps,
and generating corresponding synced data files.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Configuration
CHUNK_DURATION_SEC = 3  # Duration of each episode chunk in seconds
TARGET_FPS = 4  # Target frames per second
FRAMES_PER_CHUNK = CHUNK_DURATION_SEC * TARGET_FPS  # 12 frames per chunk


def load_dataset_info(dataset_path: Path) -> Dict:
    """Load dataset info from meta/info.json"""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info file not found: {info_path}")

    with open(info_path, 'r') as f:
        return json.load(f)


def get_available_video_keys(dataset_info: Dict) -> List[str]:
    """Extract available video keys from dataset info"""
    video_keys = []
    features = dataset_info.get('features', {})
    for key, feature_info in features.items():
        if key.startswith('observation.images.') and feature_info.get('dtype') == 'video':
            video_keys.append(key)
    return video_keys


def select_video_key(video_keys: List[str], preferred_key: Optional[str] = None) -> str:
    """Select video key to use for transformation"""
    if not video_keys:
        raise ValueError("No video keys found in dataset")
    
    if preferred_key and preferred_key in video_keys:
        return preferred_key
    
    # Default selection priority
    priority_order = ['observation.images.front', 'observation.images.image', 'observation.images.cam_high']
    
    for preferred in priority_order:
        if preferred in video_keys:
            return preferred
    
    # Return first available if no preferred found
    return video_keys[0]


def load_episode_data(dataset_path: Path, episode_idx: int) -> pd.DataFrame:
    """Load episode data from parquet file"""
    episode_file = dataset_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    if not episode_file.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_file}")

    return pd.read_parquet(episode_file)


def create_video_chunks(video_path: Path, output_dir: Path, chunk_duration: int = 3, target_fps: int = 4) -> List[Path]:
    """Create 3-second video chunks at specified fps using ffmpeg"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, get video duration
    duration_cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(video_path)
    ]

    try:
        result = subprocess.run(duration_cmd, check=True, capture_output=True, text=True)
        total_duration = float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting video duration from {video_path}: {e}")
        return []

    video_chunks = []
    chunk_idx = 0

    # Create chunks every 3 seconds
    for start_time in range(0, int(total_duration), chunk_duration):
        chunk_output = output_dir / f"chunk_{chunk_idx:03d}.mp4"

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-ss", str(start_time),  # Start time
            "-t", str(chunk_duration),  # Duration
            "-vf", f"fps={target_fps}",  # Target fps
            "-c:v", "libx264",  # Video codec
            "-preset", "fast",  # Encoding preset
            "-y",  # Overwrite existing files
            str(chunk_output)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            video_chunks.append(chunk_output)
            chunk_idx += 1
        except subprocess.CalledProcessError as e:
            print(f"Error creating chunk {chunk_idx} from {video_path}: {e}")
            continue

    return video_chunks


def create_episode_chunks(episode_data: pd.DataFrame, video_key: str, frames_per_chunk: int) -> List[Dict]:
    """Split episode data into chunks of specified frame count"""
    chunks = []
    total_frames = len(episode_data)
    
    for start_idx in range(0, total_frames, frames_per_chunk):
        end_idx = min(start_idx + frames_per_chunk, total_frames)
        chunk_data = episode_data.iloc[start_idx:end_idx].copy()
        
        # Reset index and timestamps for chunk
        chunk_data = chunk_data.reset_index(drop=True)
        chunk_data['timestamp'] = np.arange(len(chunk_data)) / TARGET_FPS
        
        chunks.append({
            'data': chunk_data,
            'start_frame': start_idx,
            'end_frame': end_idx,
            'frame_count': len(chunk_data)
        })
    
    return chunks


def save_chunk_data(chunk_data: pd.DataFrame, output_path: Path, video_key: str):
    """Save chunk data to parquet file with video key removed"""
    # Remove video key from chunk data since videos are stored separately
    if video_key in chunk_data.columns:
        chunk_data = chunk_data.drop(columns=[video_key])

    # Save to parquet
    chunk_data.to_parquet(output_path, index=False)


def copy_and_update_dataset_info(original_info: Dict, output_dir: Path, video_key: str) -> Dict:
    """Copy dataset info and update for transformed dataset"""
    new_info = original_info.copy()

    # Update features to only include selected video and non-video features
    new_features = {}
    features = original_info.get('features', {})

    for key, feature_info in features.items():
        if key == video_key:
            # Keep selected video key
            new_features[key] = feature_info.copy()
        elif not key.startswith('observation.images.'):
            # Keep non-video features (actions, states, etc.)
            new_features[key] = feature_info.copy()

    new_info['features'] = new_features
    new_info['fps'] = TARGET_FPS
    
    # Save updated info
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "info.json", 'w') as f:
        json.dump(new_info, f, indent=2)
    
    return new_info


def transform_dataset(input_dataset_path: str, output_dataset_path: str, video_key: Optional[str] = None):
    """Transform dataset by selecting video, chunking episodes, and syncing data"""
    input_path = Path(input_dataset_path)
    output_path = Path(output_dataset_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    
    print(f"Transforming dataset from {input_path} to {output_path}")
    
    # Load dataset info
    dataset_info = load_dataset_info(input_path)
    video_keys = get_available_video_keys(dataset_info)
    selected_video_key = select_video_key(video_keys, video_key)
    
    print(f"Available video keys: {video_keys}")
    print(f"Selected video key: {selected_video_key}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    data_dir = output_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy and update dataset info
    copy_and_update_dataset_info(dataset_info, output_path, selected_video_key)
    
    # Process each episode
    videos_dir = input_path / "videos" / "chunk-000"
    output_episode_idx = 0

    for episode_file in sorted((input_path / "data" / "chunk-000").glob("episode_*.parquet")):
        episode_idx = int(episode_file.stem.split('_')[1])
        print(f"Processing episode {episode_idx}...")
        
        try:
            # Load episode data
            episode_data = load_episode_data(input_path, episode_idx)
            
            # Find corresponding video file
            video_path = videos_dir / selected_video_key / f"episode_{episode_idx:06d}.mp4"
            
            if not video_path.exists():
                print(f"Warning: Video file not found: {video_path}")
                continue
            
            # Create video chunks from original video
            temp_video_dir = output_path / "temp_videos" / f"episode_{episode_idx:06d}"
            video_chunks = create_video_chunks(video_path, temp_video_dir, CHUNK_DURATION_SEC, TARGET_FPS)
            
            if not video_chunks:
                print(f"Warning: No video chunks created from {video_path}")
                continue
            
            # Create episode chunks from data
            chunks = create_episode_chunks(episode_data, selected_video_key, FRAMES_PER_CHUNK)
            
            # Save each chunk as a new episode
            for chunk_idx, chunk in enumerate(chunks):
                if chunk_idx >= len(video_chunks):
                    break  # No more video chunks available
                
                # Create video directory for this episode (matching LeRobot structure)
                episode_video_dir = output_path / "videos" / "chunk-000" / selected_video_key
                episode_video_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy video chunk to final location
                final_video_path = episode_video_dir / f"episode_{output_episode_idx:06d}.mp4"
                shutil.copy2(video_chunks[chunk_idx], final_video_path)
                
                # Save chunk data
                chunk_output_path = data_dir / f"episode_{output_episode_idx:06d}.parquet"
                save_chunk_data(chunk['data'], chunk_output_path, selected_video_key)
                
                print(f"Created episode {output_episode_idx} with 3-second video at 4fps")
                output_episode_idx += 1
            
            # Clean up temp videos
            if temp_video_dir.exists():
                shutil.rmtree(temp_video_dir)
                
        except Exception as e:
            print(f"Error processing episode {episode_idx}: {e}")
            continue
    
    print(f"Transformation complete. Created {output_episode_idx} episodes in {output_path}")


def main():
    """Main function with hardcoded paths for testing"""
    input_dataset = "/Users/michelmeyer/.cache/huggingface/lerobot/newtechmitch/o3"
    output_dataset = "transformed_dataset"
    video_key = "observation.images.front"  # Can be None for auto-selection
    
    transform_dataset(input_dataset, output_dataset, video_key)


if __name__ == "__main__":
    main()