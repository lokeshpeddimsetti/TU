import cv2
import json
#from crowd_density.crowd_sampler import CrowdSampler
from src.crowd_density.crowd_sampler import CrowdSampler
from device_scripts.dvr_input import DVRInput


CONFIG_PATH = "config/config.json"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def main():
    cfg = load_config()

    if cfg["mode"] != "rtsp":
        raise ValueError("Currently only RTSP mode supported.")

    dvr_cfg = cfg["input"]["dvr"]
    dvr = DVRInput(dvr_cfg)

    # -----------------------------------------
    # Interactive camera selection
    # -----------------------------------------
    if cfg.get("interactive", False):
        cameras = dvr.discover_cameras()

        if not cameras:
            raise RuntimeError("No active DVR channels found.")

        print("\nAvailable Cameras:")
        for idx, (ch, _) in enumerate(cameras):
            print(f"[{idx}] Channel {ch}")

        selected = int(input("Select camera index: "))
        dvr.camera_index = selected

    # -----------------------------------------
    # Open Stream
    # -----------------------------------------
    cap = dvr.open_stream()

    if not cap.isOpened():
        raise RuntimeError("Failed to open RTSP stream.")

    # -----------------------------------------
    # Initialize Crowd Sampler
    # -----------------------------------------
    analytics_cfg = cfg["analytics"]

    sampler = CrowdSampler(
        cam_name=f"camera_{dvr.camera_index}",
        sample_interval=analytics_cfg["sample_interval"],
        model_path=analytics_cfg["model_path"]
    )

    sampler.set_time_window(
        analytics_cfg["start_time"],
        analytics_cfg["stop_time"]
    )

    print("\nCrowd analytics started...\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame read failed.")
            break

        count = sampler.process_frame(frame)
        print(f"Crowd Count: {count}")

    cap.release()
    sampler.close()


if __name__ == "__main__":
    main()