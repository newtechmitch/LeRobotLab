"""Environment-specific configuration for lerobot_so_arm."""

from pathlib import Path

# Current environment - change this when switching between machines
CURRENT_ENVIRONMENT = "michel_ubuntu_local"

# Environment configurations
ENVIRONMENTS = {
    "michel_mac_local": {
        "base_path": "/Users/michelmeyer/.local/dev",
        "vjepa_root": "vjepa2-so_arm", 
        "datasets": "test_dataset",
        "models": "models",
    },
    "michel_lightning_ai": {
        "base_path": "/teamspace/studios/this_studio",
        "vjepa_root": "vjepa2",
        "datasets": "datasets", 
        "models": "models",
    },
    "michel_ubuntu_local": {
        "base_path": "/home/michelmeyer/Dev",
        "vjepa_root": "vjepa2-so_arm", 
        "datasets": "vjepa2-so_arm/datasets/test",
        "models": "vjepa2-so_arm/models",
    },
}


class EnvironmentConfig:
    """Configuration object for accessing environment-specific paths."""
    
    def __init__(self, environment: str):
        if environment not in ENVIRONMENTS:
            raise ValueError(f"Unknown environment: {environment}. Available: {list(ENVIRONMENTS.keys())}")
        
        self.environment = environment
    
    def get_path(self, key: str) -> str:
        """Get a path for the specified key."""
        config = ENVIRONMENTS[self.environment]
        
        if key not in config:
            raise ValueError(f"Unknown path key: {key}. Available: {list(config.keys())}")
        
        if key == "base_path":
            return str(Path(config["base_path"]))
        else:
            base_path = Path(config["base_path"])
            return str(base_path / config[key])
    
    def __str__(self):
        return f"EnvironmentConfig({self.environment})"


def get_current() -> str:
    """Get the currently configured environment."""
    return CURRENT_ENVIRONMENT


def get_path(key: str, environment: str = None) -> str:
    """
    Get a path for the specified key.
    
    Args:
        key: Path key (base_path, vjepa_root, datasets, models)
        environment: Environment name. If None, uses CURRENT_ENVIRONMENT.
        
    Returns:
        String path for the specified key.
    """
    if environment is None:
        environment = CURRENT_ENVIRONMENT
    
    config = EnvironmentConfig(environment)
    return config.get_path(key) 