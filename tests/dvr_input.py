import sys
import json
from pathlib import Path

# Add workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from device_scripts.dvr_input import DVRInput

with open("config/config.json") as f:
    config = json.load(f)

dvr = DVRInput(config["input"]["dvr"])
print(f"Detected decoder: {dvr.detect_decoder()}")
cameras = dvr.discover_cameras()
print(f"Found cameras: {cameras}")