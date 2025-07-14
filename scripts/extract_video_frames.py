#!/usr/bin/env python3
"""
Extract frames from MP4 video at 250ms intervals and organize into chunks of 12 images.
"""

import subprocess
import sys
from pathlib import Path

# Configuration
CHUNK_SIZE = 12  # Number of images per chunk folder
FRAME_INTERVAL_MS = 250  # Extract frame every 250ms (4fps)
VIDEO_PATH = "/Users/michelmeyer/.cache/huggingface/lerobot/newtechmitch/o3/videos/chunk-000/observation.images.front/episode_000000.mp4"
OUTPUT_DIR = "test-output"


def extract_frames(video_path: str, output_dir: str) -> None:
    """
    Extract frames from video at 250ms intervals and organize into folders of 12 images each.

    Args:
        video_path: Path to the input MP4 file
        output_dir: Output directory
    """
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"Error: Video file {video_path} does not exist")
        sys.exit(1)

    if not video_path.suffix.lower() == '.mp4':
        print(f"Warning: File {video_path} is not an MP4 file")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract frames using ffmpeg
    temp_dir = output_path / "temp_frames"
    temp_dir.mkdir(exist_ok=True)

    print(f"Extracting frames from {video_path}...")

    # Calculate fps from interval (250ms = 4fps)
    fps = 1000 // FRAME_INTERVAL_MS
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-y",  # Overwrite existing files
        str(temp_dir / "frame_%06d.png")
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    
    # Get all extracted frames
    frame_files = sorted(temp_dir.glob("frame_*.png"))
    
    if not frame_files:
        print("No frames were extracted")
        sys.exit(1)
    
    print(f"Extracted {len(frame_files)} frames")
    
    # Organize frames into chunks
    chunk_num = 0

    for i in range(0, len(frame_files), CHUNK_SIZE):
        chunk_files = frame_files[i:i + CHUNK_SIZE]
        chunk_dir = output_path / f"images-chunk-{chunk_num:02d}"
        chunk_dir.mkdir(exist_ok=True)
        
        # Move files to chunk directory with sequential naming
        for j, frame_file in enumerate(chunk_files):
            new_name = chunk_dir / f"image_{j:02d}.png"
            frame_file.rename(new_name)

        print(f"Created {chunk_dir} with {len(chunk_files)} images")
        chunk_num += 1

    # Clean up temp directory
    temp_dir.rmdir()

    print(f"Frame extraction complete. Created {chunk_num} chunk folders in {output_path}")


def main():
    extract_frames(VIDEO_PATH, OUTPUT_DIR)


if __name__ == "__main__":
    main()