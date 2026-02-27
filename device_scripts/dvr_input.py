import cv2
import subprocess
import shutil


class DVRInput:
    def __init__(self, config):
        self.ip = config["ip"]
        self.username = config["username"]
        self.password = config["password"]
        self.port = config.get("port", 554)
        self.max_channels = config.get("max_channels", 16)
        self.camera_index = 0

    # -----------------------------------------
    # Decoder Detection
    # -----------------------------------------
    def detect_decoder(self):
        if shutil.which("nvidia-smi"):
            return "NVDEC"

        try:
            subprocess.check_output(
                ["gst-inspect-1.0", "v4l2h264dec"],
                stderr=subprocess.DEVNULL
            )
            return "GSTREAMER"
        except Exception:
            pass

        return "FFMPEG"

    # -----------------------------------------
    # Discover Available Channels
    # -----------------------------------------
    def discover_cameras(self):
        cams = []
        for ch in range(1, self.max_channels + 1):
            rtsp = (
                f"rtsp://{self.username}:{self.password}@"
                f"{self.ip}:{self.port}/Streaming/Channels/{ch}01"
            )
            cap = cv2.VideoCapture(rtsp)
            if cap.isOpened():
                cams.append((ch, rtsp))
                cap.release()
        return cams

    # -----------------------------------------
    # Open Selected Stream
    # -----------------------------------------
    def open_stream(self):
        cameras = self.discover_cameras()

        if not cameras:
            raise RuntimeError("No active DVR channels found.")

        if self.camera_index >= len(cameras):
            raise IndexError("Selected camera index out of range.")

        rtsp_url = cameras[self.camera_index][1]
        decoder = self.detect_decoder()

        if decoder == "GSTREAMER":
            pipeline = (
                f"rtspsrc location={rtsp_url} latency=200 ! "
                "rtph264depay ! h264parse ! v4l2h264dec ! "
                "videoconvert ! appsink"
            )
            return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        return cv2.VideoCapture(rtsp_url)