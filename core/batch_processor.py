"""
Batch processor for multi-image classification and reporting.
"""
import os
import pandas as pd
from typing import List, Dict, Any, Callable, Optional
from .model_engine import TrafficSignModel

class BatchProcessor:
    def __init__(self, model_engine: TrafficSignModel):
        self.model_engine = model_engine

    def process_files(self, file_paths: List[str], progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, Any]]:
        results = []
        total = len(file_paths)

        for i, path in enumerate(file_paths):
            filename = os.path.basename(path)
            try:
                pred = self.model_engine.predict(path, top_k=3)
                best = pred["best"]
                results.append({
                    "Filename": filename,
                    "File Path": path,
                    "Class ID": best["class_id"],
                    "Predicted Sign": best["name"],
                    "Confidence (%)": round(best["confidence"], 2),
                    "Category": best["category"],
                    "Action Instruction": best["action_instruction"],
                    "Latency (ms)": pred["latency_ms"],
                    "Status": "Success"
                })
            except Exception as e:
                results.append({
                    "Filename": filename,
                    "File Path": path,
                    "Class ID": -1,
                    "Predicted Sign": "Error",
                    "Confidence (%)": 0.0,
                    "Category": "Error",
                    "Action Instruction": str(e),
                    "Latency (ms)": 0.0,
                    "Status": "Failed"
                })

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def export_csv(self, results: List[Dict[str, Any]], output_path: str):
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        return output_path
