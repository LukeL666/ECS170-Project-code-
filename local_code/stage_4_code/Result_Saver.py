import os
import json
import numpy as np
from local_code.base_class.result import result

class Result_Saver(result):
    def __init__(self, rName="result", rType="Save metrics", folder_path="./results"):
        self.result_name = rName
        self.result_description = rType
        self.data = None
        self.result_destination_folder_path = folder_path
        # This will automatically create nested folders like ./result/stage_4_result/
        os.makedirs(folder_path, exist_ok=True)

    def save(self, metrics, history=None):
        file_path = os.path.join(self.result_destination_folder_path, f"{self.result_name}.json")
        clean_metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'metrics': clean_metrics, 'history': history}, f, indent=4, ensure_ascii=False)
        print(f"Result saved to {file_path}")

    def load(self):
        file_path = os.path.join(self.result_destination_folder_path, f"{self.result_name}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None