# Make the utils directory a proper Python package 
from .kinematics import RobotKinematics
from .visualization import visualize_so_trajectory
from .data_loading import check_vjepa2ac_format, analyze_metadata, load_traj_from_HDF5, check_traj_format
from .transform import DatasetTransformer
from .calibration import detect_so_calibration 