"""
lerobot_so_arm: Utilities and tools for SO100/SO101 robot arm trajectory analysis.

This package provides:
- Robot kinematics and trajectory processing
- Data loading and transformation utilities  
- Visualization tools for 3D trajectory analysis
- Environment configuration management
"""

__version__ = "0.1.0"

# Make key utilities easily accessible
from .config import get_path, get_current
from .utils.kinematics import RobotKinematics
from .utils.data_loading import (
    load_traj_from_HDF5,
    check_traj_format,
    check_vjepa2ac_format,
    analyze_metadata,
    load_so_episode
)

__all__ = [
    'get_path',
    'get_current', 
    'RobotKinematics',
    'load_traj_from_HDF5',
    'check_traj_format',
    'check_vjepa2ac_format',
    'analyze_metadata',
    'load_so_episode'
] 