'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
import numpy as np

class Setting_Train_Test_Split(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        # use pre-split train/test from dataset (no random split in stage 3)
        X_train, y_train = loaded_data['train']['X'], loaded_data['train']['y']
        X_test, y_test = loaded_data['test']['X'], loaded_data['test']['y']

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.fold_count = 0
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None