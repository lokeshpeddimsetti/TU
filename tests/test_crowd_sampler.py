import json
import os
import time
from datetime import datetime, timedelta
import sys

# ensure source package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest

# --- create lightweight stubs for external packages before importing module under test ---
class _StubYOLO:
    def __init__(self, path):
        self.path = path
    def to(self, device):
        self.device = device
    def fuse(self):
        pass
    def predict(self, frame, **kwargs):
        if frame is None:
            return [type("R", (), {"boxes": None})()]
        return [type("R", (), {"boxes": type("B", (), {"xyxy": [[0,0,30,30]]})()})()]

# stub ultralytics module
ultralytics = type(sys)('ultralytics')
ultralytics.YOLO = _StubYOLO
sys.modules['ultralytics'] = ultralytics

# stub torch module with cuda availability
torch = type(sys)('torch')
torch.cuda = type('C', (), {'is_available': staticmethod(lambda: False)})
sys.modules['torch'] = torch

from crowd_density.crowd_sampler import CrowdSampler


class DummyResults:
    def __init__(self, boxes):
        self.boxes = boxes


class DummyBoxes:
    def __init__(self, xyxy_list):
        self.xyxy = xyxy_list


class DummyYOLO:
    def __init__(self, path):
        # ignore model path
        self.path = path

    def to(self, device):
        self.device = device

    def fuse(self):
        pass

    def predict(self, frame, **kwargs):
        # return single detection box for non-empty frame, else no boxes
        if frame is None:
            return [DummyResults(None)]
        # create a box object with tolist
        class Box:
            def __init__(self, coords):
                self.coords = coords
            def tolist(self):
                return self.coords
        box = Box([0, 0, 30, 30])
        return [DummyResults(DummyBoxes([box]))]


@pytest.fixture(autouse=True)
def patch_yolo(monkeypatch):
    monkeypatch.setattr("crowd_density.crowd_sampler.YOLO", DummyYOLO)


def read_log(path):
    with open(path, "r") as f:
        return json.load(f)


def test_sampling_and_logging(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    cs = CrowdSampler("cam1", sample_interval=0.1, json_dir=str(log_dir))

    # first frame should produce count 1
    count = cs.process_frame(frame=object())
    assert count == 1

    # second call within interval returns last_count without predicting
    count2 = cs.process_frame(frame=object())
    assert count2 == 1

    # wait beyond sample_interval and call again
    time.sleep(0.11)
    count3 = cs.process_frame(frame=object())
    assert count3 == 1

    # check that JSON file exists and contains entries
    data = read_log(log_dir / "crowd_counts.json")
    assert "cam1" in data
    today = datetime.now().strftime("%Y-%m-%d")
    assert today in data["cam1"]
    assert len(data["cam1"][today]) >= 2


def test_time_window(tmp_path):
    cs = CrowdSampler("cam2", json_dir=str(tmp_path))
    # set window so that current time is outside
    now = datetime.now()
    start = (now + timedelta(hours=1)).strftime("%H:%M")
    stop = (now + timedelta(hours=2)).strftime("%H:%M")
    cs.set_time_window(start, stop)
    assert cs.process_frame(frame=object()) == 0

    # set window in past so hard stop triggers
    start = (now - timedelta(hours=2)).strftime("%H:%M")
    stop = (now - timedelta(hours=1)).strftime("%H:%M")
    cs.set_time_window(start, stop)
    assert cs.process_frame(frame=object()) == 0


if __name__ == "__main__":
    pytest.main([__file__])
