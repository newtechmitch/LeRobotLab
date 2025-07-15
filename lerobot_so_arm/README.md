# lerobot_so_arm Installation

## Installation

Install in development mode. 

Note: as the original V-JEPA 2 project cannot be installed in development mode on macOS due to the `decord` video processing library, which doesn't have pre-built wheels for Apple Silicon, we commented decord in the requirements.txt, and did a custom setup script below.

Run the custom setup script from the project root:

```bash
# From the project root directory (vjepa2-so_arm/)
python lerobot_so_arm_setup.py develop
```

### Verify Installation

```bash
python -c "from lerobot_so_arm.config import get_path; print('Installation successful!')"
```


### If you are not on Mac

If you are not on macOS, you will need to install decord for the main V-JEPA 2 project.
