import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.spatial.transform import Rotation

# Try to import plotly for 3D visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. 3D visualizations will be disabled.")

# Try to import ipywidgets for interactive controls
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("Warning: ipywidgets not available. Interactive controls will be disabled.")


def compute_joint_positions_in_world_coordinates(joint_positions, robot_kinematics):
    """
    Compute the positions of all robot frames using the forward_kinematics method
    
    Args:
        joint_positions: Array of 6 joint positions [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        robot_kinematics: RobotKinematics instance
        
    Returns:
        Dictionary of frame positions and orientations
    """
    # The issue is that forward_kinematics applies coordinate transformations
    # For visualization, we want to use the measurements directly but apply joint rotations
    # Let's use a hybrid approach: measurements for positions, but apply rotations manually
    
    from .kinematics import screw_axis_to_transform
    
    # Convert to radians for screw axis calculations
    robot_pos_rad = np.deg2rad(joint_positions[:5])
    
    # Build cumulative transforms like the original manual approach
    X_BS = screw_axis_to_transform(robot_kinematics.S_BS, robot_pos_rad[0])
    X_BH = X_BS @ screw_axis_to_transform(robot_kinematics.S_BH, robot_pos_rad[1]) 
    X_BF = X_BH @ screw_axis_to_transform(robot_kinematics.S_BF, robot_pos_rad[2])
    X_BR = X_BF @ screw_axis_to_transform(robot_kinematics.S_BR, robot_pos_rad[3])
    X_BG = X_BR @ screw_axis_to_transform(robot_kinematics.S_BG, robot_pos_rad[4])
    
    # Use measurements as initial positions (these are the correct joint locations)
    initial_positions = {
        'base': robot_kinematics.measurements['base'],
        'shoulder': robot_kinematics.measurements['shoulder'],
        'humerus': robot_kinematics.measurements['humerus'],
        'forearm': robot_kinematics.measurements['forearm'],
        'wrist': robot_kinematics.measurements['wrist'],
        'gripper': robot_kinematics.measurements['gripper'],
    }
    
    # Apply transforms to get actual positions
    frame_transforms = {
        'base': np.eye(4),  # Base doesn't move
        'shoulder': X_BS,
        'humerus': X_BH,
        'forearm': X_BF,
        'wrist': X_BR,
        'gripper': X_BG,
    }
    
    frame_data = {}
    
    # Compute positions for each frame
    for frame in ['base', 'shoulder', 'humerus', 'forearm', 'wrist', 'gripper']:
        # Apply cumulative transform to initial position
        initial_pos_homo = np.array(initial_positions[frame] + [1])
        transformed_pos_homo = frame_transforms[frame] @ initial_pos_homo
        position = transformed_pos_homo[:3]
        
        # Get orientation from transform
        orientation = frame_transforms[frame][:3, :3]
        
        # Create full 4x4 transform
        transform = np.eye(4)
        transform[:3, :3] = orientation
        transform[:3, 3] = position
        
        # Store in result dictionary
        frame_data[frame] = {
            'position': position,
            'orientation': orientation,
            'transform': transform
        }
    
    # Add gripper_tip (offset from gripper)
    gripper_tip_offset = np.array([0.12, 0, 0, 1])  # 12cm forward from gripper
    gripper_initial_pos = np.array(initial_positions['gripper'] + [1])
    gripper_tip_pos_homo = X_BG @ (gripper_initial_pos + gripper_tip_offset - np.array([0, 0, 0, 1]))
    gripper_tip_position = gripper_tip_pos_homo[:3]
    
    frame_data['gripper_tip'] = {
        'position': gripper_tip_position,
        'orientation': X_BG[:3, :3],
        'transform': transform
    }
    
    return frame_data


def load_trajectory_data_from_episode(episode_path):
    """
    Load trajectory data from an episode directory.
    
    Args:
        episode_path (str): Path to the episode directory containing trajectory.h5
        
    Returns:
        dict: Dictionary containing cartesian_positions and gripper_positions, or None if loading fails
    """
    trajectory_path = os.path.join(episode_path, 'trajectory.h5')
    
    if not os.path.exists(trajectory_path):
        print(f"ERROR: Trajectory file not found at {trajectory_path}")
        return None
    
    try:
        with h5py.File(trajectory_path, 'r') as h5f:
            # Check if the file has the expected structure
            if ('observation' in h5f and 
                'robot_state' in h5f['observation'] and 
                'cartesian_position' in h5f['observation']['robot_state']):
                
                # Load cartesian positions
                cartesian_positions = np.array(h5f['observation']['robot_state']['cartesian_position'])
                
                # Load gripper positions if available
                gripper_positions = None
                if 'gripper_position' in h5f['observation']['robot_state']:
                    gripper_positions = np.array(h5f['observation']['robot_state']['gripper_position'])
                
                # Load joint states if available
                joint_states = None
                if 'state' in h5f['observation']:
                    joint_states = np.array(h5f['observation']['state'])
                
                # Load camera extrinsics if available
                camera_extrinsics = {}
                if 'camera_extrinsics' in h5f['observation']:
                    for camera_name in h5f['observation']['camera_extrinsics'].keys():
                        camera_extrinsics[camera_name] = np.array(h5f['observation']['camera_extrinsics'][camera_name])
                        print(f"Found camera extrinsics for: {camera_name}")
                
                return {
                    'cartesian_positions': cartesian_positions,
                    'gripper_positions': gripper_positions,
                    'joint_states': joint_states,
                    'camera_extrinsics': camera_extrinsics
                }
            else:
                print(f"ERROR: Trajectory file does not have the expected structure")
                return None
    except Exception as e:
        print(f"ERROR: Failed to load trajectory data: {e}")
        return None


def visualize_trajectory_3d(trajectory_data, title="Robot End-Effector Trajectory", 
                            show_coord_system=True, show_robot_at_time=None, 
                            robot_type="so_new_calibration"):
    """
    Create an interactive 3D visualization of the robot trajectory using Plotly.
    
    Args:
        trajectory_data (dict): Dictionary containing 'cartesian_positions' and optionally 'gripper_positions', 'joint_states'
        title (str): Title for the plot
        show_coord_system (bool): Whether to show coordinate system axes at origin
        show_robot_at_time (int, optional): Time index to show robot configuration
        robot_type (str): Robot type for kinematics
        
    Returns:
        plotly.graph_objects.Figure: The 3D trajectory plot, or None if plotly not available
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Cannot create 3D visualization.")
        return None
        
    if trajectory_data is None or 'cartesian_positions' not in trajectory_data:
        print("No trajectory data available for visualization")
        return None
    
    # Extract position data (XYZ)
    positions = trajectory_data['cartesian_positions']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    
    # Extract gripper positions if available
    gripper_positions = trajectory_data.get('gripper_positions', None)
    
    # Create a color gradient based on time
    time_values = np.linspace(0, 1, len(x))
    
    # Create hover text for trajectory line
    hover_text = []
    for i in range(len(x)):
        text = (f"Gripper Tip:<br>X: {x[i]:.4f} m<br>"
                f"Y: {y[i]:.4f} m<br>Z: {z[i]:.4f} m")
        if gripper_positions is not None:
            text += f"<br>Gripper: {gripper_positions[i, 0]:.4f}"
        text += f"<br>Time: {i}/{len(x)-1}"
        hover_text.append(text)
    
    # Create the 3D figure
    fig = go.Figure()
    
    # Add the trajectory line
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        line=dict(color=time_values, colorscale='Viridis', width=4),
        mode='lines',
        name='Trajectory',
        hovertemplate='%{text}<extra></extra>',
        text=hover_text
    ))
    
    # Create hover text for start point
    start_text = (f"START<br>Gripper Tip:<br>X: {x[0]:.4f} m<br>"
                  f"Y: {y[0]:.4f} m<br>Z: {z[0]:.4f} m")
    if gripper_positions is not None:
        start_text += f"<br>Gripper: {gripper_positions[0, 0]:.4f}"
    
    # Add markers for start and end points
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode='markers',
        marker=dict(size=8, color='green'),
        name='Start Point',
        hovertemplate='%{text}<extra></extra>',
        text=[start_text]
    ))
    
    # Create hover text for end point
    end_text = (f"END<br>Gripper Tip:<br>X: {x[-1]:.4f} m<br>"
                f"Y: {y[-1]:.4f} m<br>Z: {z[-1]:.4f} m")
    if gripper_positions is not None:
        end_text += f"<br>Gripper: {gripper_positions[-1, 0]:.4f}"
    
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='End Point',
        hovertemplate='%{text}<extra></extra>',
        text=[end_text]
    ))
    
    # Add markers at regular intervals
    step = max(1, len(x) // 20)  # Show about 20 markers along the path
    
    # Create hover text for waypoints
    waypoint_hover_text = []
    for i in range(0, len(x), step):
        text = f"WAYPOINT {i//step + 1}<br>Gripper Tip:<br>X: {x[i]:.4f} m<br>Y: {y[i]:.4f} m<br>Z: {z[i]:.4f} m"
        if gripper_positions is not None:
            text += f"<br>Gripper: {gripper_positions[i, 0]:.4f}"
        text += f"<br>Time: {i}/{len(x)-1}"
        waypoint_hover_text.append(text)
    
    fig.add_trace(go.Scatter3d(
        x=x[::step], y=y[::step], z=z[::step],
        mode='markers',
        marker=dict(
            size=5,
            color=time_values[::step],
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Waypoints',
        hovertemplate='%{text}<extra></extra>',
        text=waypoint_hover_text
    ))
    
    # Add coordinate system at origin if requested
    if show_coord_system:
        fig = add_base_coordinate_system(fig, scale=0.1)
    
    # Add cameras if available
    if 'camera_extrinsics' in trajectory_data and trajectory_data['camera_extrinsics']:
        fig = add_camera_visualization(fig, trajectory_data['camera_extrinsics'], scale=0.05)
        print(f"Added {len(trajectory_data['camera_extrinsics'])} camera(s) to visualization")
    
    # Add robot configuration if requested
    if show_robot_at_time is not None and 'joint_states' in trajectory_data:
        fig = add_robot_configuration(fig, trajectory_data['joint_states'], 
                                     time_index=show_robot_at_time, robot_type=robot_type)
    
    # Update the layout for better visualization
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Z Position (m)',
            aspectmode='data'  # 'cube', 'data', 'manual'
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig


def add_robot_configuration(fig, joint_states, time_index=0, robot_type="so_new_calibration", scale=1.0):
    """
    Add robot link visualization to the 3D plot for a specific time.
    
    Args:
        fig: The plotly figure to add the robot to
        joint_states: Array of joint states over time
        time_index: Which time step to visualize
        robot_type: Robot type for kinematics
        scale: Scale factor for visualization
        
    Returns:
        The updated figure
    """
    if not PLOTLY_AVAILABLE:
        return fig
        
    if joint_states is None or len(joint_states) == 0:
        print("No joint states available for robot visualization")
        return fig
        
    if time_index >= len(joint_states):
        print(f"Time index {time_index} out of range (max: {len(joint_states)-1})")
        return fig
    
    try:
        from .kinematics import RobotKinematics
        
        # Initialize kinematics
        robot_kinematics = RobotKinematics(robot_type)
        
        # Get joint positions at the specified time (first 5 joints)
        joint_pos = joint_states[time_index][:5]
        
        # Debug: print joint values to check if they're reasonable
        print(f"Joint positions at time {time_index}: {joint_pos}")
        
        # If joint values are very large, they might already be in degrees
        # If they're small (< 10), they might be in radians and need conversion
        if np.max(np.abs(joint_pos)) < 10:
            joint_pos = np.rad2deg(joint_pos)
            print(f"Converted to degrees: {joint_pos}")
        
        # Define the robot links in order from base to tip
        frames = ["base", "shoulder", "humerus", "forearm", "wrist", "gripper_tip"]
        
        # Compute positions for each frame
        positions = {}
        for frame in frames:
            pose = robot_kinematics.forward_kinematics(joint_pos, frame=frame)
            positions[frame] = pose[:3, 3]  # Extract position (x, y, z)
        
        # Define connections between frames
        connections = [
            ("base", "shoulder"),
            ("shoulder", "humerus"), 
            ("humerus", "forearm"),
            ("forearm", "wrist"),
            ("wrist", "gripper_tip")
        ]
        
        # Colors for different segments
        segment_colors = {
            "base": "black",
            "shoulder": "red", 
            "humerus": "orange",
            "forearm": "yellow",
            "wrist": "green",
            "gripper_tip": "blue"
        }
        
        # Add robot segments
        for start_frame, end_frame in connections:
            start_pos = positions[start_frame]
            end_pos = positions[end_frame]
            
            fig.add_trace(go.Scatter3d(
                x=[start_pos[0], end_pos[0]],
                y=[start_pos[1], end_pos[1]], 
                z=[start_pos[2], end_pos[2]],
                mode='lines',
                line=dict(color=segment_colors[start_frame], width=8),
                name=f"Robot {start_frame}-{end_frame}",
                hovertemplate=f"Robot segment: {start_frame} to {end_frame}<extra></extra>",
                showlegend=False
            ))
        
        # Add joint markers
        for frame in frames:
            pos = positions[frame]
            fig.add_trace(go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]], 
                z=[pos[2]],
                mode='markers',
                marker=dict(size=6, color=segment_colors[frame]),
                name=f"Robot {frame}",
                hovertemplate=f"Robot {frame}<br>X: {pos[0]:.3f}m<br>Y: {pos[1]:.3f}m<br>Z: {pos[2]:.3f}m<extra></extra>",
                showlegend=False
            ))
            
        print(f"Added robot configuration at time {time_index}")
        
    except Exception as e:
        print(f"Error adding robot configuration: {e}")
    
    return fig


def add_base_coordinate_system(fig, scale=0.1):
    """
    Add a coordinate system at the origin (0,0,0) to represent the robot's base frame.
    
    Args:
        fig: The plotly figure to add the coordinate system to
        scale (float): The size of the coordinate axes
        
    Returns:
        The updated figure
    """
    if not PLOTLY_AVAILABLE:
        return fig
        
    # Remove the small axis indicators - they are now handled in the interactive function
    # Only keep this function for compatibility with other parts of the code
    # The interactive function has its own coordinate system implementation
    
    return fig


def add_camera_visualization(fig, camera_extrinsics, scale=0.05):
    """
    Add camera visualization to the 3D plot.
    
    Args:
        fig: The plotly figure to add cameras to
        camera_extrinsics (dict): Dictionary of camera_name -> pose [x,y,z,rx,ry,rz]
        scale (float): Size of the camera coordinate system
        
    Returns:
        The updated figure
    """
    if not PLOTLY_AVAILABLE or not camera_extrinsics:
        return fig
    
    for camera_name, pose in camera_extrinsics.items():
        # Extract position and rotation
        position = pose[:3]
        euler_angles = pose[3:6]  # XYZ euler angles in radians
        
        # Convert to rotation matrix
        rotation_matrix = Rotation.from_euler('xyz', euler_angles, degrees=False).as_matrix()
        
        # Create camera coordinate axes
        axes_length = scale
        
        # Camera coordinate system vectors
        x_axis = rotation_matrix @ np.array([axes_length, 0, 0])
        y_axis = rotation_matrix @ np.array([0, axes_length, 0])
        z_axis = rotation_matrix @ np.array([0, 0, axes_length])
        
        # Add camera position marker
        fig.add_trace(go.Scatter3d(
            x=[position[0]],
            y=[position[1]],
            z=[position[2]],
            mode='markers',
            marker=dict(size=8, color='orange', symbol='diamond'),
            name=f'Camera {camera_name}',
            hovertemplate=f'<b>Camera: {camera_name}</b><br>' +
                         f'X: {position[0]:.3f}m<br>' +
                         f'Y: {position[1]:.3f}m<br>' +
                         f'Z: {position[2]:.3f}m<br>' +
                         '<extra></extra>'
        ))
        
        # Add camera X-axis (red)
        fig.add_trace(go.Scatter3d(
            x=[position[0], position[0] + x_axis[0]],
            y=[position[1], position[1] + x_axis[1]],
            z=[position[2], position[2] + x_axis[2]],
            mode='lines',
            line=dict(color='red', width=4),
            name=f'{camera_name} X',
            showlegend=False
        ))
        
        # Add camera Y-axis (green)
        fig.add_trace(go.Scatter3d(
            x=[position[0], position[0] + y_axis[0]],
            y=[position[1], position[1] + y_axis[1]],
            z=[position[2], position[2] + y_axis[2]],
            mode='lines',
            line=dict(color='green', width=4),
            name=f'{camera_name} Y',
            showlegend=False
        ))
        
        # Add camera Z-axis (blue) - this points in the camera's viewing direction
        fig.add_trace(go.Scatter3d(
            x=[position[0], position[0] + z_axis[0]],
            y=[position[1], position[1] + z_axis[1]],
            z=[position[2], position[2] + z_axis[2]],
            mode='lines',
            line=dict(color='blue', width=4),
            name=f'{camera_name} Z (view)',
            showlegend=False
        ))
        
        # Add camera label
        fig.add_trace(go.Scatter3d(
            x=[position[0] + 0.02],
            y=[position[1] + 0.02],
            z=[position[2] + 0.02],
            mode='text',
            text=[camera_name],
            textposition='middle center',
            textfont=dict(color='orange', size=12),
            name=f'{camera_name} label',
            showlegend=False
        ))
    
    return fig


def visualize_trajectory_components(trajectory_data, title="Robot Trajectory Components"):
    """
    Create 2D time series plots of the trajectory components (position, orientation, gripper).
    
    Args:
        trajectory_data (dict): Dictionary containing 'cartesian_positions' and optionally 'gripper_positions'
        title (str): Title for the plot
        
    Returns:
        plotly.graph_objects.Figure: The component plots, or None if plotly not available
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Cannot create component visualization.")
        return None
        
    if trajectory_data is None or 'cartesian_positions' not in trajectory_data:
        print("No trajectory data available for visualization")
        return None
    
    # Extract data
    positions = trajectory_data['cartesian_positions']
    time = np.arange(len(positions))
    
    # Create three subplots: position, orientation, and gripper
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=("Position (XYZ)", "Orientation (RPY)", "Gripper State"),
                        vertical_spacing=0.1,
                        shared_xaxes=True)
    
    # Add position traces (XYZ)
    fig.add_trace(go.Scatter(x=time, y=positions[:, 0], mode='lines', name='X Position', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=positions[:, 1], mode='lines', name='Y Position', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=positions[:, 2], mode='lines', name='Z Position', line=dict(color='blue')), row=1, col=1)
    
    # Add orientation traces (Roll, Pitch, Yaw)
    fig.add_trace(go.Scatter(x=time, y=positions[:, 3], mode='lines', name='Roll', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=positions[:, 4], mode='lines', name='Pitch', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=positions[:, 5], mode='lines', name='Yaw', line=dict(color='blue')), row=2, col=1)
    
    # Add gripper state if available
    if trajectory_data['gripper_positions'] is not None:
        gripper = trajectory_data['gripper_positions'].flatten()
        fig.add_trace(go.Scatter(x=time, y=gripper, mode='lines', name='Gripper', line=dict(color='purple')), row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title=title,
        height=900,
        width=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Angle (rad)", row=2, col=1)
    fig.update_yaxes(title_text="Gripper State", row=3, col=1)
    
    # Update x-axis label
    fig.update_xaxes(title_text="Time Step", row=3, col=1)
    
    return fig


def list_available_episodes(dataset_path):
    """
    List all available episodes in the dataset directory.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        list: Sorted list of episode names that contain trajectory.h5 files
    """
    episodes = []
    if os.path.exists(dataset_path):
        for item in os.listdir(dataset_path):
            episode_path = os.path.join(dataset_path, item)
            if os.path.isdir(episode_path) and os.path.exists(os.path.join(episode_path, 'trajectory.h5')):
                episodes.append(item)
    
    return sorted(episodes)


def print_trajectory_statistics(trajectory_data):
    """
    Print statistics about the trajectory data.
    
    Args:
        trajectory_data (dict): Dictionary containing trajectory data
    """
    if trajectory_data is None or 'cartesian_positions' not in trajectory_data:
        print("No trajectory data available")
        return
        
    cartesian = trajectory_data['cartesian_positions']
    print(f"Cartesian positions shape: {cartesian.shape}")
    
    if trajectory_data['gripper_positions'] is not None:
        print(f"Gripper positions shape: {trajectory_data['gripper_positions'].shape}")
    
    print("\nCartesian Position Statistics:")
    print(f"X range: [{np.min(cartesian[:, 0]):.4f}, {np.max(cartesian[:, 0]):.4f}] meters")
    print(f"Y range: [{np.min(cartesian[:, 1]):.4f}, {np.max(cartesian[:, 1]):.4f}] meters")
    print(f"Z range: [{np.min(cartesian[:, 2]):.4f}, {np.max(cartesian[:, 2]):.4f}] meters")
    
    print("\nOrientation Statistics:")
    print(f"Roll range: [{np.min(cartesian[:, 3]):.4f}, {np.max(cartesian[:, 3]):.4f}] radians")
    print(f"Pitch range: [{np.min(cartesian[:, 4]):.4f}, {np.max(cartesian[:, 4]):.4f}] radians")
    print(f"Yaw range: [{np.min(cartesian[:, 5]):.4f}, {np.max(cartesian[:, 5]):.4f}] radians")


def visualize_so_trajectory(joint_states, timestamps=None, title="SO Robot Trajectory"):
    """
    Visualize SO100/SO101 robot trajectory data with both joint plots and 3D trajectory.
    
    Args:
        joint_states (np.ndarray): Array of shape (T, 6) containing joint positions
                                  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        timestamps (np.ndarray, optional): Array of shape (T,) containing timestamps.
                                          If None, an array of indices will be created.
        title (str, optional): Title for the visualization.
        
    Returns:
        dict: Dictionary containing statistics about the trajectory and 3D positions
    """
    if not isinstance(joint_states, np.ndarray) or len(joint_states) == 0:
        raise ValueError("Joint states data is empty or invalid")
        
    print(f"Using joint states with shape: {joint_states.shape}")
    
    # Create timestamps if not provided
    if timestamps is None or not isinstance(timestamps, np.ndarray):
        timestamps = np.arange(len(joint_states))
        print(f"Created timestamps array with shape: {timestamps.shape}")
    else:
        print(f"Using timestamps with shape: {timestamps.shape}")
    
    # Extract components from joint states
    # For SO100/SO101, the joint states are:
    # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    shoulder_pan = joint_states[:, 0]
    shoulder_lift = joint_states[:, 1]
    elbow_flex = joint_states[:, 2]
    wrist_flex = joint_states[:, 3]
    wrist_roll = joint_states[:, 4]
    gripper = joint_states[:, 5]
    
    # Create four separate plots: 3 for joint data and 1 for 3D trajectory
    if PLOTLY_AVAILABLE:
        # Create subplots with 3D plot
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Shoulder & Elbow Joints', 'Wrist Joints', 'Gripper Position', '3D Robot Trajectory'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "scene"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Joint positions (first 3 joints)
        fig.add_trace(go.Scatter(x=timestamps, y=shoulder_pan, name='Shoulder Pan', 
                                line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=timestamps, y=shoulder_lift, name='Shoulder Lift', 
                                line=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=timestamps, y=elbow_flex, name='Elbow Flex', 
                                line=dict(color='blue')), row=1, col=1)
        
        # Plot 2: Wrist joint positions
        fig.add_trace(go.Scatter(x=timestamps, y=wrist_flex, name='Wrist Flex', 
                                line=dict(color='red')), row=1, col=2)
        fig.add_trace(go.Scatter(x=timestamps, y=wrist_roll, name='Wrist Roll', 
                                line=dict(color='green')), row=1, col=2)
        
        # Plot 3: Gripper position
        fig.add_trace(go.Scatter(x=timestamps, y=gripper, name='Gripper', 
                                line=dict(color='black')), row=2, col=1)
        
        # Plot 4: 3D trajectory using forward kinematics
        try:
            from .kinematics import RobotKinematics
            
            robot_kinematics = RobotKinematics('so_new_calibration')
            
            # Compute end-effector positions for all time steps
            end_effector_positions = []
            gripper_positions_3d = []
            
            print("Computing 3D trajectory using corrected forward kinematics...")
            
            # Define coordinate transformation matrix
            R = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ])
            
            for i, joints in enumerate(joint_states):
                # Use corrected forward kinematics
                frame_data = compute_joint_positions_in_world_coordinates(joints, robot_kinematics)
                
                # Get gripper tip position and apply transformation
                end_effector_pos = frame_data['gripper_tip']['position']
                end_effector_pos_transformed = R @ end_effector_pos
                
                gripper_pos = frame_data['gripper']['position']
                gripper_pos_transformed = R @ gripper_pos
                
                end_effector_positions.append(end_effector_pos_transformed)
                gripper_positions_3d.append(gripper_pos_transformed)
            
            end_effector_positions = np.array(end_effector_positions)
            gripper_positions_3d = np.array(gripper_positions_3d)
            
            # Create 3D trajectory plot
            # End-effector trajectory
            fig.add_trace(go.Scatter3d(
                x=end_effector_positions[:, 0],
                y=end_effector_positions[:, 1],
                z=end_effector_positions[:, 2],
                mode='lines+markers',
                line=dict(color='blue', width=4),
                marker=dict(size=2),
                name='End-Effector Trajectory',
                hovertemplate='Position: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'
            ), row=2, col=2)
            
            # Add start and end markers
            fig.add_trace(go.Scatter3d(
                x=[end_effector_positions[0, 0]],
                y=[end_effector_positions[0, 1]],
                z=[end_effector_positions[0, 2]],
                mode='markers',
                marker=dict(size=8, color='green'),
                name='Start',
                hovertemplate='Start: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'
            ), row=2, col=2)
            
            fig.add_trace(go.Scatter3d(
                x=[end_effector_positions[-1, 0]],
                y=[end_effector_positions[-1, 1]],
                z=[end_effector_positions[-1, 2]],
                mode='markers',
                marker=dict(size=8, color='red'),
                name='End',
                hovertemplate='End: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'
            ), row=2, col=2)
            
            # Add robot configuration at start position
            start_frame_data = compute_joint_positions_in_world_coordinates(joint_states[0], robot_kinematics)
            frame_names = ['base', 'shoulder', 'humerus', 'forearm', 'wrist', 'gripper', 'gripper_tip']
            
            # Extract positions for kinematic chain
            chain_positions = []
            for frame in frame_names:
                if frame in start_frame_data:
                    chain_positions.append(start_frame_data[frame]['position'])
            
            chain_positions = np.array(chain_positions)
            
            # Add kinematic chain visualization
            fig.add_trace(go.Scatter3d(
                x=chain_positions[:, 0],
                y=chain_positions[:, 1],
                z=chain_positions[:, 2],
                mode='lines+markers',
                line=dict(color='gray', width=6),
                marker=dict(size=6, color='orange'),
                name='Robot Chain (Start)',
                hovertemplate='Frame: %{text}<br>Position: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>',
                text=frame_names
            ), row=2, col=2)
            
        except ImportError:
            print("Could not import kinematics module for 3D visualization")
            fig.add_annotation(
                text="3D Visualization requires kinematics module",
                x=0.5, y=0.5,
                showarrow=False,
                row=2, col=2
            )
        except Exception as e:
            print(f"Error computing 3D trajectory: {e}")
            fig.add_annotation(
                text=f"Error: {str(e)}",
                x=0.5, y=0.5,
                showarrow=False,
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        
        fig.update_yaxes(title_text="Joint Position (rad)", row=1, col=1)
        fig.update_yaxes(title_text="Joint Position (rad)", row=1, col=2)
        fig.update_yaxes(title_text="Gripper Position", row=2, col=1)
        
        # Update 3D scene
        fig.update_scenes(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data',
            row=2, col=2
        )
        
        fig.show()
        
    else:
        # Fallback to matplotlib if plotly not available
        fig, axs = plt.subplots(3, 1, figsize=(14, 15))
        
        # Plot 1: Joint positions (first 3 joints)
        axs[0].plot(timestamps, shoulder_pan, 'r-', label='Shoulder Pan')
        axs[0].plot(timestamps, shoulder_lift, 'g-', label='Shoulder Lift')
        axs[0].plot(timestamps, elbow_flex, 'b-', label='Elbow Flex')
        axs[0].set_title('Robot Joint Positions (Shoulder and Elbow)')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Joint Position (rad)')
        axs[0].grid(True)
        axs[0].legend()
        
        # Plot 2: Wrist joint positions
        axs[1].plot(timestamps, wrist_flex, 'r-', label='Wrist Flex')
        axs[1].plot(timestamps, wrist_roll, 'g-', label='Wrist Roll')
        axs[1].set_title('Robot Wrist Joint Positions')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Joint Position (rad)')
        axs[1].grid(True)
        axs[1].legend()
        
        # Plot 3: Gripper position
        axs[2].plot(timestamps, gripper, 'k-', label='Gripper')
        axs[2].set_title('Robot Gripper Position')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Gripper Position')
        axs[2].grid(True)
        axs[2].legend()
        
        # Set y-axis limits for gripper plot
        min_val = np.min(gripper)
        max_val = np.max(gripper)
        margin = 0.1 * (max_val - min_val)
        axs[2].set_ylim([min_val - margin, max_val + margin])
        
        plt.tight_layout()
        plt.show()
    
    # Calculate statistics
    stats = {
        "duration": len(joint_states),
        "joint_ranges": {
            "shoulder_pan": (np.min(shoulder_pan), np.max(shoulder_pan)),
            "shoulder_lift": (np.min(shoulder_lift), np.max(shoulder_lift)),
            "elbow_flex": (np.min(elbow_flex), np.max(elbow_flex)),
            "wrist_flex": (np.min(wrist_flex), np.max(wrist_flex)),
            "wrist_roll": (np.min(wrist_roll), np.max(wrist_roll)),
            "gripper": (np.min(gripper), np.max(gripper))
        }
    }
    
    # Add 3D trajectory statistics if computed
    if PLOTLY_AVAILABLE and 'end_effector_positions' in locals():
        stats["3d_trajectory"] = {
            "start_position": end_effector_positions[0].tolist(),
            "end_position": end_effector_positions[-1].tolist(),
            "total_distance": np.sum(np.linalg.norm(np.diff(end_effector_positions, axis=0), axis=1)),
            "workspace_bounds": {
                "x": (np.min(end_effector_positions[:, 0]), np.max(end_effector_positions[:, 0])),
                "y": (np.min(end_effector_positions[:, 1]), np.max(end_effector_positions[:, 1])),
                "z": (np.min(end_effector_positions[:, 2]), np.max(end_effector_positions[:, 2]))
            }
        }
    
    # Print statistics
    print("\nTrajectory Statistics:")
    print(f"Total duration: {stats['duration']} timesteps")
    print(f"Joint angle ranges:")
    print(f"  Shoulder Pan: [{stats['joint_ranges']['shoulder_pan'][0]:.4f}, {stats['joint_ranges']['shoulder_pan'][1]:.4f}] rad")
    print(f"  Shoulder Lift: [{stats['joint_ranges']['shoulder_lift'][0]:.4f}, {stats['joint_ranges']['shoulder_lift'][1]:.4f}] rad")
    print(f"  Elbow Flex: [{stats['joint_ranges']['elbow_flex'][0]:.4f}, {stats['joint_ranges']['elbow_flex'][1]:.4f}] rad")
    print(f"  Wrist Flex: [{stats['joint_ranges']['wrist_flex'][0]:.4f}, {stats['joint_ranges']['wrist_flex'][1]:.4f}] rad")
    print(f"  Wrist Roll: [{stats['joint_ranges']['wrist_roll'][0]:.4f}, {stats['joint_ranges']['wrist_roll'][1]:.4f}] rad")
    print(f"  Gripper: [{stats['joint_ranges']['gripper'][0]:.4f}, {stats['joint_ranges']['gripper'][1]:.4f}]")
    
    if "3d_trajectory" in stats:
        print(f"\n3D Trajectory Statistics:")
        print(f"  Start position: [{stats['3d_trajectory']['start_position'][0]:.3f}, {stats['3d_trajectory']['start_position'][1]:.3f}, {stats['3d_trajectory']['start_position'][2]:.3f}] m")
        print(f"  End position: [{stats['3d_trajectory']['end_position'][0]:.3f}, {stats['3d_trajectory']['end_position'][1]:.3f}, {stats['3d_trajectory']['end_position'][2]:.3f}] m")
        print(f"  Total distance: {stats['3d_trajectory']['total_distance']:.3f} m")
        print(f"  Workspace bounds:")
        print(f"    X: [{stats['3d_trajectory']['workspace_bounds']['x'][0]:.3f}, {stats['3d_trajectory']['workspace_bounds']['x'][1]:.3f}] m")
        print(f"    Y: [{stats['3d_trajectory']['workspace_bounds']['y'][0]:.3f}, {stats['3d_trajectory']['workspace_bounds']['y'][1]:.3f}] m")
        print(f"    Z: [{stats['3d_trajectory']['workspace_bounds']['z'][0]:.3f}, {stats['3d_trajectory']['workspace_bounds']['z'][1]:.3f}] m")
    
    return stats


def visualize_trajectory_with_robot_interactive(trajectory_data, title="Robot Trajectory with Interactive Configuration", robot_type="so_new_calibration"):
    """
    Create an interactive 3D visualization with a time slider to show robot configuration.
    
    Args:
        trajectory_data (dict): Dictionary containing trajectory data with 'joint_states'
        title (str): Title for the plot
        robot_type (str): Robot type for kinematics
    
    Returns:
        ipywidgets interactive display
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Cannot create 3D visualization.")
        return None
        
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available. Cannot create interactive controls.")
        return None
        
    if trajectory_data is None or 'joint_states' not in trajectory_data:
        print("No joint states available for robot visualization")
        return None
    
    joint_states = trajectory_data['joint_states']
    if joint_states is None or len(joint_states) == 0:
        print("Joint states data is empty")
        return None
    
    # Create the base figure once
    base_fig = visualize_trajectory_3d(
        trajectory_data, 
        title=title,
        show_coord_system=True,
        show_robot_at_time=None,  # Don't add robot initially
        robot_type=robot_type
    )
    
    if base_fig is None:
        print("Failed to create base visualization")
        return None
    
    # Convert to FigureWidget for dynamic updates
    fig_widget = go.FigureWidget(base_fig)
    
    # Set fixed scene bounds and uirevision to prevent auto-resizing
    # Define workspace bounds that accommodate the robot's full range and axis labels
    workspace_bounds = {
        'x': [-1.0, 1.1],
        'y': [-1.0, 1.1], 
        'z': [-0.5, 1.5]
    }
    
    fig_widget.update_layout(
        uirevision='constant',
        scene=dict(
            xaxis=dict(range=workspace_bounds['x'], autorange=False),
            yaxis=dict(range=workspace_bounds['y'], autorange=False),
            zaxis=dict(range=workspace_bounds['z'], autorange=False),
            aspectmode='cube'  # Keep proportions
        )
    )
    
    # Add fixed reference points to establish scene bounds
    fig_widget.add_scatter3d(
        x=[workspace_bounds['x'][0], workspace_bounds['x'][1], 0, 0],
        y=[workspace_bounds['y'][0], workspace_bounds['y'][1], 0, 0],
        z=[workspace_bounds['z'][0], workspace_bounds['z'][1], 0, 0],
        mode='markers',
        marker=dict(size=1, color='rgba(0,0,0,0)'),  # Invisible markers
        name='Scene bounds',
        showlegend=False,
        hoverinfo='skip'
    )
    
    # Add dotted coordinate axes from -1 to 1 with arrows and labels
    # X-axis (red dotted line)
    fig_widget.add_scatter3d(
        x=[-1, 1],
        y=[0, 0],
        z=[0, 0],
        mode='lines',
        line=dict(color='red', width=3, dash='dot'),
        name='X-axis',
        showlegend=False,
        hoverinfo='skip'
    )
    # X-axis arrow
    fig_widget.add_scatter3d(
        x=[0.9, 1, 0.9],
        y=[0, 0, 0],
        z=[-0.05, 0, 0.05],
        mode='lines',
        line=dict(color='red', width=3),
        name='X-arrow',
        showlegend=False,
        hoverinfo='skip'
    )
    # X-axis label (red)
    fig_widget.add_scatter3d(
        x=[1.05],
        y=[0],
        z=[0],
        mode='text',
        text=['X'],
        textposition='middle center',
        textfont=dict(color='red', size=16),
        name='X-label',
        showlegend=False,
        hoverinfo='skip'
    )
    
    # Y-axis (green dotted line)
    fig_widget.add_scatter3d(
        x=[0, 0],
        y=[-1, 1],
        z=[0, 0],
        mode='lines',
        line=dict(color='green', width=3, dash='dot'),
        name='Y-axis',
        showlegend=False,
        hoverinfo='skip'
    )
    # Y-axis arrow
    fig_widget.add_scatter3d(
        x=[0, 0, 0],
        y=[0.9, 1, 0.9],
        z=[-0.05, 0, 0.05],
        mode='lines',
        line=dict(color='green', width=3),
        name='Y-arrow',
        showlegend=False,
        hoverinfo='skip'
    )
    # Y-axis label (green)
    fig_widget.add_scatter3d(
        x=[0],
        y=[1.05],
        z=[0],
        mode='text',
        text=['Y'],
        textposition='middle center',
        textfont=dict(color='green', size=16),
        name='Y-label',
        showlegend=False,
        hoverinfo='skip'
    )
    
    # Z-axis (blue dotted line)
    fig_widget.add_scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[-1, 1],
        mode='lines',
        line=dict(color='blue', width=3, dash='dot'),
        name='Z-axis',
        showlegend=False,
        hoverinfo='skip'
    )
    # Z-axis arrow
    fig_widget.add_scatter3d(
        x=[-0.05, 0, 0.05],
        y=[0, 0, 0],
        z=[0.9, 1, 0.9],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Z-arrow',
        showlegend=False,
        hoverinfo='skip'
    )
    # Z-axis label (blue)
    fig_widget.add_scatter3d(
        x=[0],
        y=[0],
        z=[1.05],
        mode='text',
        text=['Z'],
        textposition='middle center',
        textfont=dict(color='blue', size=16),
        name='Z-label',
        showlegend=False,
        hoverinfo='skip'
    )
    
    # Create time step slider
    time_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(joint_states) - 1,
        step=1,
        description='Time Step:',
        style={'description_width': 'initial'},
        layout={'width': '80%'}
    )
    
    # Store robot traces for efficient updating
    robot_traces = {'segments': [], 'markers': []}
    
    def update_robot_configuration(time_step):
        """Update robot configuration by modifying existing traces data"""
        try:
            from .kinematics import RobotKinematics
            
            # Compute new robot positions
            robot_kinematics = RobotKinematics(robot_type)
            
            # Define frames and connections
            frames = ["base", "shoulder", "humerus", "forearm", "wrist", "gripper_tip"]
            connections = [
                ("base", "shoulder"), ("shoulder", "humerus"), 
                ("humerus", "forearm"), ("forearm", "wrist"), ("wrist", "gripper_tip")
            ]
            segment_colors = {
                "base": "black", "shoulder": "red", "humerus": "orange",
                "forearm": "yellow", "wrist": "green", "gripper_tip": "blue"
            }
            
            # Compute positions using the corrected function
            frame_data = compute_joint_positions_in_world_coordinates(joint_states[time_step], robot_kinematics)
            positions = {}
            for frame in frames:
                if frame in frame_data:
                    positions[frame] = frame_data[frame]['position']
            
            # Initialize robot traces if this is the first call
            if not robot_traces['segments']:
                # Add robot segments
                for i, (start_frame, end_frame) in enumerate(connections):
                    start_pos = positions[start_frame]
                    end_pos = positions[end_frame]
                    
                    trace = go.Scatter3d(
                        x=[start_pos[0], end_pos[0]],
                        y=[start_pos[1], end_pos[1]], 
                        z=[start_pos[2], end_pos[2]],
                        mode='lines',
                        line=dict(color=segment_colors[start_frame], width=8),
                        name=f"Robot {start_frame}-{end_frame}",
                        hovertemplate=f"Robot segment: {start_frame} to {end_frame}<extra></extra>",
                        showlegend=False
                    )
                    fig_widget.add_trace(trace)
                    robot_traces['segments'].append(len(fig_widget.data) - 1)
                
                # Add joint markers
                for i, frame in enumerate(frames):
                    pos = positions[frame]
                    trace = go.Scatter3d(
                        x=[pos[0]], y=[pos[1]], z=[pos[2]],
                        mode='markers',
                        marker=dict(size=6, color=segment_colors[frame]),
                        name=f"Robot {frame}",
                        hovertemplate=f"Robot {frame}<br>X: {pos[0]:.3f}m<br>Y: {pos[1]:.3f}m<br>Z: {pos[2]:.3f}m<extra></extra>",
                        showlegend=False
                    )
                    fig_widget.add_trace(trace)
                    robot_traces['markers'].append(len(fig_widget.data) - 1)
            else:
                # Update existing traces
                # Update segments
                for i, (start_frame, end_frame) in enumerate(connections):
                    start_pos = positions[start_frame]
                    end_pos = positions[end_frame]
                    trace_idx = robot_traces['segments'][i]
                    
                    with fig_widget.batch_update():
                        fig_widget.data[trace_idx].x = [start_pos[0], end_pos[0]]
                        fig_widget.data[trace_idx].y = [start_pos[1], end_pos[1]]
                        fig_widget.data[trace_idx].z = [start_pos[2], end_pos[2]]
                
                # Update markers
                for i, frame in enumerate(frames):
                    pos = positions[frame]
                    trace_idx = robot_traces['markers'][i]
                    
                    with fig_widget.batch_update():
                        fig_widget.data[trace_idx].x = [pos[0]]
                        fig_widget.data[trace_idx].y = [pos[1]]
                        fig_widget.data[trace_idx].z = [pos[2]]
            
            # Update only the title (without affecting camera)
            fig_widget.update_layout(title=f"{title} - Time Step: {time_step}")
            
        except Exception as e:
            print(f"Error updating robot configuration: {e}")
    
    # Connect slider to update function
    def on_slider_change(change):
        if change['name'] == 'value':
            update_robot_configuration(change['new'])
    
    time_slider.observe(on_slider_change, names='value')
    
    # Display the controls and figure
    display(widgets.VBox([time_slider, fig_widget]))
    
    # Initialize with first time step
    update_robot_configuration(0)
    
    return fig_widget 