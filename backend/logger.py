import json
import os
import time
from typing import Dict


def now_ns() -> int:
    return time.time_ns()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def open_log(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "a", encoding="utf-8")


class EventLogger:
    def __init__(self, base_dir: str, experiment_mode: str):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"exp_{experiment_mode}", f"run_{ts}")
        ensure_dir(self.run_dir)
        self.request_fp = open_log(os.path.join(self.run_dir, "request.log"))
        self.stage_fp = open_log(os.path.join(self.run_dir, "stage.log"))
        self.pipeline_fp = open_log(os.path.join(self.run_dir, "pipeline.log"))
        self.system_fp = open_log(os.path.join(self.run_dir, "system.log"))

    def log(self, fp, payload: Dict) -> None:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        fp.flush()

    def log_request(self, payload: Dict) -> None:
        self.log(self.request_fp, payload)

    def log_stage(self, payload: Dict) -> None:
        self.log(self.stage_fp, payload)

    def log_pipeline(self, payload: Dict) -> None:
        self.log(self.pipeline_fp, payload)

    def log_system(self, payload: Dict) -> None:
        self.log(self.system_fp, payload)

    def close(self) -> None:
        self.request_fp.close()
        self.stage_fp.close()
        self.pipeline_fp.close()
        self.system_fp.close()
