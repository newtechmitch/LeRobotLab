import os
import json
import h5py
import numpy as np
from typing import Dict, Tuple, Any


def load_traj_from_HDF5(trajectory_path: str) -> Dict[str, np.ndarray]:
    """
    Load trajectory data from an HDF5 file.
    
    Args:
        trajectory_path: Path to the trajectory.h5 file
        
    Returns:
        Dictionary containing trajectory data with keys:
        - joint_states: Joint positions from /observation/state (if available)
        - cartesian_position: End-effector positions from /observation/robot_state/cartesian_position (if available)
        - gripper_position: Gripper positions from /observation/robot_state/gripper_position (if available)
        - timestamps: Timestamps from /metadata/timestamp (if available)
        - actions: Actions from /action/data (if available)
    """
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(f"Trajectory file not found at {trajectory_path}")
    
    with h5py.File(trajectory_path, 'r') as h5file:
        # Check for data availability
        has_joint_states = 'observation' in h5file and 'state' in h5file['observation']
        has_timestamps = 'metadata' in h5file and 'timestamp' in h5file['metadata']
        has_actions = 'action' in h5file and 'data' in h5file['action']
        has_cartesian = ('observation' in h5file and 'robot_state' in h5file['observation'] and
                         'cartesian_position' in h5file['observation']['robot_state'])
        has_gripper = ('observation' in h5file and 'robot_state' in h5file['observation'] and
                       'gripper_position' in h5file['observation']['robot_state'])
        
        # Load available data
        trajectory_data = {
            'joint_states': np.array(h5file['observation']['state']) if has_joint_states else None,
            'cartesian_position': (np.array(h5file['observation']['robot_state']['cartesian_position'])
                                   if has_cartesian else None),
            'gripper_position': (np.array(h5file['observation']['robot_state']['gripper_position'])
                                 if has_gripper else None),
            'timestamps': np.array(h5file['metadata']['timestamp']) if has_timestamps else None,
            'actions': np.array(h5file['action']['data']) if has_actions else None
        }
        
        return trajectory_data


def check_traj_format(trajectory_data: Dict[str, np.ndarray], trajectory_path: str = None) -> None:
    """
    Check and display information about trajectory data format.
    
    Args:
        trajectory_data: Dictionary containing trajectory data
        trajectory_path: Optional path for display purposes
    """
    print("****** Checking trajectory format ******")
    if trajectory_path:
        print(f"  Path: {os.path.basename(trajectory_path)}")
    
    # Check data availability
    has_joint_states = trajectory_data['joint_states'] is not None
    has_cartesian = trajectory_data['cartesian_position'] is not None
    has_gripper = trajectory_data['gripper_position'] is not None
    has_timestamps = trajectory_data['timestamps'] is not None
    has_actions = trajectory_data['actions'] is not None
       
    # Show data shapes with key paths
    if has_joint_states:
        print(f"  Joint states [observation/state]: {trajectory_data['joint_states'].shape}")
    if has_cartesian:
        print(f"  Cartesian positions [observation/robot_state/cartesian_position]: "
              f"{trajectory_data['cartesian_position'].shape}")
    if has_gripper:
        print(f"  Gripper positions [observation/robot_state/gripper_position]: "
              f"{trajectory_data['gripper_position'].shape}")
    if has_timestamps:
        print(f"  Timestamps [metadata/timestamp]: {trajectory_data['timestamps'].shape}")
    if has_actions:
        print(f"  Actions [action/data]: {trajectory_data['actions'].shape}")


def check_vjepa2ac_format(path: str) -> Tuple[bool, str]:
    """
    Check if a folder follows the SO100/SO101 format.
    
    Args:
        path: Path to the folder to check
        
    Returns:
        Tuple of (is_valid, message) where is_valid is a boolean indicating if the format is valid,
        and message contains details about the validation.
    """
    if not os.path.exists(path):
        return False, f"Path {path} does not exist."
    
    if not os.path.isdir(path):
        return False, f"Path {path} is not a directory."
    
    # Check for required files
    required_files = ["metadata.json", "trajectory.h5"]
    missing_files = []
    
    for f in required_files:
        if not os.path.isfile(os.path.join(path, f)):
            missing_files.append(f)
    
    # Check for recordings/MP4 directory
    recordings_mp4_path = os.path.join(path, "recordings", "MP4")
    has_recordings_mp4 = os.path.isdir(recordings_mp4_path)
    
    # Report missing components
    if missing_files or not has_recordings_mp4:
        message = "Missing required components: "
        if missing_files:
            message += f"Files: {', '.join(missing_files)}. "
        if not has_recordings_mp4:
            message += "Directory: recordings/MP4."
        return False, message
    
    # Check metadata.json structure
    try:
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Check for basic metadata fields
        required_metadata_keys = ["episode_id", "episode_name", "source_dataset"]
        missing_keys = [k for k in required_metadata_keys if k not in metadata]
        
        if missing_keys:
            return False, f"metadata.json is missing required keys: {', '.join(missing_keys)}"
        
        # Check trajectory.h5 file structure
        trajectory_file = os.path.join(path, "trajectory.h5")
        
        try:
            with h5py.File(trajectory_file, "r") as f:
                # Check for basic structure expected in SO100/SO101 datasets
                expected_groups = ['observation', 'metadata', 'action']
                missing_groups = [g for g in expected_groups if g not in f]
                
                if missing_groups:
                    return False, f"trajectory.h5 is missing required groups: {', '.join(missing_groups)}"
                
                # Check for observation/state
                if 'state' not in f['observation']:
                    return False, "trajectory.h5 is missing observation/state data"
                
                # Check for metadata/timestamp
                if 'timestamp' not in f['metadata']:
                    return False, "trajectory.h5 is missing metadata/timestamp data"
                
                # Check for action/data
                if 'data' not in f['action']:
                    return False, "trajectory.h5 is missing action/data"
                
        except Exception as e:
            return False, f"Error reading trajectory.h5 file: {str(e)}"
                
        return True, "Directory follows SO100/SO101 format."
        
    except Exception as e:
        return False, f"Error validating metadata.json: {str(e)}"


def analyze_metadata(path: str) -> str:
    """
    Load metadata, check structure, and provide detailed information about it.
    
    Args:
        path: Path to the SO100/SO101 dataset folder
        
    Returns:
        String with detailed information about the metadata and dataset structure
    """
    if not os.path.exists(path):
        return f"Path {path} does not exist."
    
    metadata_path = os.path.join(path, "metadata.json")
    if not os.path.isfile(metadata_path):
        return f"Metadata file not found at {metadata_path}"
    
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        info = []
        info.append("SO100/SO101 Dataset Analysis")
        info.append("--------------------------")
        
        # Basic information
        if "episode_id" in metadata:
            info.append(f"Episode ID: {metadata['episode_id']}")
        
        if "episode_name" in metadata:
            info.append(f"Episode Name: {metadata['episode_name']}")
            
        if "source_dataset" in metadata:
            info.append(f"Source Dataset: {metadata['source_dataset']}")
            
        if "total_frames" in metadata:
            info.append(f"Total Frames: {metadata['total_frames']}")
            
        if "duration_seconds" in metadata:
            info.append(f"Duration: {metadata['duration_seconds']} seconds")
            
        if "fps" in metadata:
            info.append(f"FPS: {metadata['fps']}")
        
        # Data keys information
        if "data_keys" in metadata:
            info.append("\nData Keys:")
            for key, value in metadata["data_keys"].items():
                info.append(f"- {key}: {value}")
        
        # Files information
        if "files" in metadata:
            info.append("\nFiles:")
            for key, value in metadata["files"].items():
                info.append(f"- {key}: {value}")
        
        # Trajectory analysis
        trajectory_path = os.path.join(path, "trajectory.h5")
        if os.path.isfile(trajectory_path):
            info.append("\nTrajectory Analysis:")
            with h5py.File(trajectory_path, "r") as f:
                info.append(f"File: {trajectory_path}")
                
                # List all top-level groups in the file
                top_groups = list(f.keys())
                info.append(f"Top-level groups: {top_groups}")
                
                # Check for observation/state data
                if 'observation' in f and 'state' in f['observation']:
                    state_data = f['observation']['state']
                    info.append("\nJoint States:")
                    info.append(f"- Shape: {state_data.shape}")
                    info.append(f"- Type: {state_data.dtype}")
                    
                    if len(state_data.shape) > 0 and state_data.shape[0] > 0:
                        if state_data.shape[1] == 6:
                            components = "shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper"
                            info.append(f"- Components: [{components}]")
                            
                            # Sample values
                            first_frame = np.array(state_data[0])
                            info.append(f"- First frame: {first_frame}")
                
                # Check for metadata/timestamp
                if 'metadata' in f and 'timestamp' in f['metadata']:
                    timestamp_data = f['metadata']['timestamp']
                    info.append("\nTimestamps:")
                    info.append(f"- Shape: {timestamp_data.shape}")
                    info.append(f"- Type: {timestamp_data.dtype}")
                    
                    if len(timestamp_data) > 0:
                        duration = timestamp_data[-1] - timestamp_data[0]
                        info.append(f"- Duration: {duration} seconds")
                
                # Check for action data
                if 'action' in f and 'data' in f['action']:
                    action_data = f['action']['data']
                    info.append("\nActions:")
                    info.append(f"- Shape: {action_data.shape}")
                    info.append(f"- Type: {action_data.dtype}")
        
        # Recordings analysis
        recordings_path = os.path.join(path, "recordings", "MP4")
        if os.path.isdir(recordings_path):
            mp4_files = [f for f in os.listdir(recordings_path) if f.endswith('.mp4')]
            info.append("\nRecordings:")
            info.append(f"- Number of MP4 files: {len(mp4_files)}")
            if mp4_files:
                camera_names = [os.path.splitext(f)[0] for f in mp4_files]
                info.append(f"- Available cameras: {', '.join(camera_names)}")
        
        # Additional metadata fields
        excluded_keys = ["episode_id", "episode_name", "source_dataset", 
                        "total_frames", "duration_seconds", "fps", 
                        "data_keys", "files"]
        other_keys = [k for k in metadata.keys() if k not in excluded_keys]
        if other_keys:
            info.append("\nAdditional Metadata Fields:")
            for key in other_keys:
                info.append(f"- {key}: {metadata[key]}")
        
        return "\n".join(info)
        
    except Exception as e:
        return f"Error analyzing metadata: {str(e)}"


def load_so_episode(episode_path: str) -> Dict[str, Any]:
    """
    Load a single SO100/SO101 episode from its directory path.
    
    Args:
        episode_path: Path to episode directory containing metadata.json, trajectory.h5, and recordings/MP4/
        
    Returns:
        Dict containing episode data: metadata, trajectory_data, video_path, and episode info
    """
    if not os.path.exists(episode_path):
        raise FileNotFoundError(f"Episode path does not exist: {episode_path}")
    
    episode_data = {
        'path': episode_path,
        'name': os.path.basename(episode_path),
        'metadata': None,
        'trajectory_data': None,
        'video_path': None
    }
    
    # Load metadata
    metadata_path = os.path.join(episode_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            episode_data['metadata'] = json.load(f)
    
    # Load trajectory data
    trajectory_path = os.path.join(episode_path, 'trajectory.h5')
    if os.path.exists(trajectory_path):
        episode_data['trajectory_data'] = load_traj_from_HDF5(trajectory_path)
    
    # Find video file
    video_dir = os.path.join(episode_path, 'recordings', 'MP4')
    if os.path.exists(video_dir):
        video_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
        if video_files:
            episode_data['video_path'] = os.path.join(video_dir, video_files[0])
    
    return episode_data 