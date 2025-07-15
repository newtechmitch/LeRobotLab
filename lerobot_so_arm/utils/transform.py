#!/usr/bin/env python3
"""Dataset transformer for V-JEPA 2-AC format."""

import os
import sys
import h5py
import numpy as np
import shutil
import json
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation

# Add project root to path before importing local modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from lerobot_so_arm.utils.kinematics import RobotKinematics
from lerobot_so_arm.utils.data_loading import load_traj_from_HDF5
from lerobot_so_arm.utils.calibration import detect_so_calibration


class DatasetTransformer:
    """Transforms SO100/SO101 episodes to V-JEPA 2-AC format with automatic calibration detection."""
    
    def __init__(self):
        """Initialize DatasetTransformer with automatic calibration detection."""
        self._kinematics_cache = {}
    
    def _get_robot_kinematics(self, calibration_name: str) -> RobotKinematics:
        """Get cached RobotKinematics instance for the given calibration."""
        if calibration_name not in self._kinematics_cache:
            self._kinematics_cache[calibration_name] = RobotKinematics(calibration_name)
        return self._kinematics_cache[calibration_name]
    
    def compute_cartesian_positions(self, joint_states: np.ndarray, creation_date=None):
        """
        Convert joint positions to cartesian positions using forward kinematics.
        
        Automatically detects the appropriate calibration for this trajectory.
        """
        T = joint_states.shape[0]
        cartesian_positions = np.zeros((T, 6))
        gripper_positions = np.zeros((T, 1))
        
        # Detect calibration type for this trajectory
        calibration_name = detect_so_calibration(joint_states, creation_date)
        print(f"Auto-detected calibration: {calibration_name}")
        
        # Get appropriate robot kinematics
        robot_kinematics = self._get_robot_kinematics(calibration_name)
        
        for t in range(T):
            ee_pose = robot_kinematics.forward_kinematics(joint_states[t, :5])
            cartesian_positions[t, :3] = ee_pose[:3, 3]
            rpy = Rotation.from_matrix(ee_pose[:3, :3]).as_euler('zyx', degrees=False) #convert with the proper convention and in radians
            cartesian_positions[t, 3:] = rpy
            gripper_positions[t, 0] = joint_states[t, 5] / 100.0
        
        return cartesian_positions, gripper_positions
    
    def transform_trajectory_file(self, source_path: str, output_path: str):
        """Transform trajectory.h5 to include cartesian positions."""
        trajectory_data = load_traj_from_HDF5(source_path)
        joint_states = trajectory_data['joint_states']
        
        if joint_states is None:
            raise ValueError(f"No joint states in {source_path}")
        
        cartesian_positions, gripper_positions = self.compute_cartesian_positions(joint_states)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with h5py.File(output_path, 'w') as h5f:
            # Create groups
            obs_group = h5f.create_group('observation')
            meta_group = h5f.create_group('metadata')
            
            # Save original joint states in observation/state
            obs_group.create_dataset('state', data=joint_states, dtype=np.float64)
            
            # Add robot_state for V-JEPA 2-AC
            robot_state_group = obs_group.create_group('robot_state')
            robot_state_group.create_dataset('cartesian_position', data=cartesian_positions, dtype=np.float64)
            robot_state_group.create_dataset('gripper_position', data=gripper_positions, dtype=np.float64)
            
            # Save timestamps if available
            if trajectory_data['timestamps'] is not None:
                meta_group.create_dataset('timestamp', data=trajectory_data['timestamps'], dtype=np.float64)
    
    def transform_episode(self, source_path: str, output_path: str):
        """Transform complete episode."""
        os.makedirs(output_path, exist_ok=True)
        
        # Copy metadata
        source_metadata = os.path.join(source_path, 'metadata.json')
        if os.path.exists(source_metadata):
            shutil.copy2(source_metadata, os.path.join(output_path, 'metadata.json'))
        
        # Transform trajectory
        source_trajectory = os.path.join(source_path, 'trajectory.h5')
        output_trajectory = os.path.join(output_path, 'trajectory.h5')
        self.transform_trajectory_file(source_trajectory, output_trajectory)
        
        # Copy recordings
        source_recordings = os.path.join(source_path, 'recordings')
        if os.path.exists(source_recordings):
            shutil.copytree(source_recordings, os.path.join(output_path, 'recordings'), dirs_exist_ok=True)
    
    def transform_dataset(self, source_folder: str, output_folder: str, dataset_list_file: str = 'dataset_list.txt'):
        """Transform complete dataset."""
        # Read episode list
        with open(os.path.join(source_folder, dataset_list_file), 'r') as f:
            episode_paths = [os.path.join(source_folder, line.strip()) for line in f if line.strip()]
        
        os.makedirs(output_folder, exist_ok=True)
        
        for episode_path in tqdm(episode_paths, desc="Transforming episodes"):
            episode_name = os.path.basename(episode_path)
            output_path = os.path.join(output_folder, episode_name)
            try:
                self.transform_episode(episode_path, output_path)
            except Exception as e:
                print(f"Error transforming {episode_name}: {e}")

 