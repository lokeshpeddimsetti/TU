#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
from datetime import datetime
from ultralytics import YOLO
import torch


class CrowdSampler:
    """
    Frame-driven, stateful crowd sampler
    - Uses shared frames from main loop
    - Runs YOLO only every sample_interval seconds
    - Always returns last known crowd count
    - Writes crowd data to JSON continuously
    """

    def __init__(
        self,
        cam_name,
        sample_interval=5,
        model_path="models/head-based_detection.pt",
        json_dir="logs"
    ):
        self.cam_name = cam_name
        self.sample_interval = sample_interval
        self.last_sample_time = 0
        self.last_count = 0

        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model.fuse()

        print(f"[CrowdSampler] {cam_name} | device={self.device}")

        self.start_time = None
        self.stop_time = None

        # JSON logging
        os.makedirs(json_dir, exist_ok=True)
        self.json_path = os.path.join(json_dir, "crowd_counts.json")
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w") as f:
                json.dump({}, f, indent=4)

    # --------------------------------------------------
    def _log_json(self, count):
        now = datetime.now()
        day = now.strftime("%Y-%m-%d")
        entry = {
            "time": now.strftime("%H:%M:%S"),
            "count": count
        }

        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

        data.setdefault(self.cam_name, {})
        data[self.cam_name].setdefault(day, [])
        data[self.cam_name][day].append(entry)

        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=4)

    def set_time_window(self, start_str, stop_str):
        today = datetime.now().date()

        self.start_time = datetime.strptime(
            f"{today} {start_str}", "%Y-%m-%d %H:%M"
        )
        self.stop_time = datetime.strptime(
            f"{today} {stop_str}", "%Y-%m-%d %H:%M"
        )

    def process_frame(self, frame):
        now_dt = datetime.now()

        # ⛔ NOT STARTED YET
        if self.start_time and now_dt < self.start_time:
            return 0

        # ⛔ HARD STOP
        if self.stop_time and now_dt >= self.stop_time:
            return 0

        # ⏱ sampling clock (FIX)
        now = time.time()

        if now - self.last_sample_time < self.sample_interval:
            return self.last_count

        self.last_sample_time = now

        try:
            results = self.model.predict(
                frame,
                classes=[0],      # HEAD class
                conf=0.35,
                iou=0.4,
                verbose=False,
                device=self.device
            )

            count = 0
            MIN_BOX_SIZE = 20  # pixels

            if results and results[0].boxes is not None:
                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = box.tolist()
                    w = x2 - x1
                    h = y2 - y1

                    if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                        continue

                    count += 1

            self.last_count = count

        except Exception as e:
            print(f"[CrowdSampler] inference error: {e}")

        self._log_json(self.last_count)
        return self.last_count

    # --------------------------------------------------
    def finalize_hours(self, force=False):
        """
        Kept for compatibility with your pipeline.
        You can extend this later for hourly peaks.
        """
        return []

    # --------------------------------------------------
    def close(self):
        pass
