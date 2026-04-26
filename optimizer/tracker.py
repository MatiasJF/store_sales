"""
Experiment tracking: logs every config tried and its score.
"""

import json
import time
from config import OUTPUT_DIR


class ExperimentTracker:
    def __init__(self):
        self.experiments: list[dict] = []
        self.start_time = time.time()

    def log(self, experiment: dict):
        experiment["timestamp"] = time.time() - self.start_time
        self.experiments.append(experiment)

    def best(self) -> dict | None:
        valid = [e for e in self.experiments if e.get("score", float("inf")) < float("inf")]
        if not valid:
            return None
        return min(valid, key=lambda x: x["score"])

    def save(self):
        path = OUTPUT_DIR / "experiments.json"
        # Convert non-serializable types
        clean = []
        for exp in self.experiments:
            entry = {}
            for k, v in exp.items():
                if isinstance(v, float) and (v == float("inf") or v != v):
                    entry[k] = str(v)
                else:
                    entry[k] = v
            clean.append(entry)
        with open(path, "w") as f:
            json.dump(clean, f, indent=2, default=str)

    def summary(self) -> str:
        best = self.best()
        total = len(self.experiments)
        elapsed = time.time() - self.start_time
        if best:
            return (
                f"Experiments: {total} | Best RMSLE: {best['score']:.5f} | "
                f"Model: {best.get('model_name', '?')} | "
                f"Elapsed: {elapsed:.0f}s"
            )
        return f"Experiments: {total} | No valid results | Elapsed: {elapsed:.0f}s"
