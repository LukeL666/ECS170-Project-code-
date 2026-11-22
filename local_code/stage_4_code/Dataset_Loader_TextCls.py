import os
import numpy as np
from local_code.base_class.dataset import dataset

class Dataset_Loader_TextCls(dataset):
    def __init__(self, data_root_path=None, glove_util=None, max_len=200, dName="TextClassification", dDescription="RNN Text Classification Dataset"):
        # Manual assignment to avoid super() module conflict
        self.dataset_name = dName
        self.dataset_descrition = dDescription 
        self.data = None
        
        self.data_root_path = data_root_path
        self.glove = glove_util
        self.max_len = max_len

    def _read_split(self, split_name):
        split_path = os.path.join(self.data_root_path, split_name)
        texts = []
        labels = []
        
        # Positive samples
        pos_path = os.path.join(split_path, 'pos')
        if os.path.exists(pos_path):
            for filename in os.listdir(pos_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(pos_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        tokens = f.read().strip().lower().split()
                        texts.append(tokens)
                        labels.append(1)
                        
        # Negative samples
        neg_path = os.path.join(split_path, 'neg')
        if os.path.exists(neg_path):
            for filename in os.listdir(neg_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(neg_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        tokens = f.read().strip().lower().split()
                        texts.append(tokens)
                        labels.append(0)
                        
        return texts, labels

    def load(self):
        print(f"Loading data from: {self.data_root_path}")
        train_texts, train_labels = self._read_split('train')
        test_texts, test_labels = self._read_split('test')

        print("Encoding texts to token IDs...")
        train_X = np.array([self.glove.encode(t, self.max_len) for t in train_texts], dtype=np.int64)
        train_y = np.array(train_labels, dtype=np.int64)
        
        test_X = np.array([self.glove.encode(t, self.max_len) for t in test_texts], dtype=np.int64)
        test_y = np.array(test_labels, dtype=np.int64)

        # Shuffle training data
        train_idx = np.random.permutation(len(train_X))
        train_X = train_X[train_idx]
        train_y = train_y[train_idx]

        self.data = {
            'train_X': train_X, 'train_y': train_y,
            'test_X': test_X, 'test_y': test_y
        }
        print(f"Data loaded. Train size: {len(train_X)}, Test size: {len(test_X)}")
        return self.data