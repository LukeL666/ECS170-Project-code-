import os
import csv
import re
import numpy as np
from local_code.base_class.dataset import dataset

class Dataset_Loader_TextGen(dataset):
    def __init__(self, file_path=None, glove_util=None, max_len=30, dName="TextGeneration", dDescription="RNN Joke Generation Dataset"):
        # Manual assignment
        self.dataset_name = dName
        self.dataset_descrition = dDescription
        self.data = None
        
        self.file_path = file_path
        self.glove = glove_util
        self.max_len = max_len

    def load(self):
        print("Loading Text Data...")
        texts = []      
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    text = row[1]
                    tokens = re.findall(r"[\w']+|[.,!?;]", text.lower())
                    if len(tokens) > 3:
                        texts.append(tokens)
        
        inputs, targets = [], []
        for tokens in texts:
            encoded = self.glove.encode(tokens, self.max_len)
            inputs.append(encoded[:-1])
            targets.append(encoded[1:])
            
        self.data = {
            'train_X': np.array(inputs, dtype=np.int64),
            'train_y': np.array(targets, dtype=np.int64),
        }
        print(f"Loaded {len(texts)} texts.")
        return self.data