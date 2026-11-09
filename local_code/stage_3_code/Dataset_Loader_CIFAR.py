from local_code.base_class.dataset import dataset
import numpy as np
import pickle
import os


class Dataset_Loader_CIFAR(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = 'CIFAR'   # pickle file name

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _pack(self, instance_list):
        """list of {'image','label'} -> X (N,3,32,32), y (N,)"""
        X = np.array([inst['image'] for inst in instance_list], dtype=np.float32)
        y = np.array([inst['label'] for inst in instance_list], dtype=np.int64)

        # make sure shape is (N,3,32,32)
        if X.ndim == 4:
            if X.shape[-1] == 3:          # (N,32,32,3)  HWC -> NCHW
                X = X.transpose(0, 3, 1, 2)
            # else already (N,3,32,32), do nothing
        elif X.ndim == 2:                 # (N, 3072) flattened in C,H,W order
            X = X.reshape(-1, 3, 32, 32)

        # normalize to [0,1] if values look like 0-255
        if X.max() > 1.0:
            X = X / 255.0
        return X, y

    def load(self):
        print('loading CIFAR data...')
        path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        f = open(path, 'rb')
        raw = pickle.load(f)
        f.close()

        X_train, y_train = self._pack(raw['train'])
        X_test,  y_test  = self._pack(raw['test'])

        self.data = {
            'train': {'X': X_train, 'y': y_train},
            'test':  {'X': X_test,  'y': y_test},
        }
        print('CIFAR loaded: train', X_train.shape, 'test', X_test.shape)
        return self.data