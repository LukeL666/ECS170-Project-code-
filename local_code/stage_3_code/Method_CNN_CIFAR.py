from local_code.base_class.method import method
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class Method_CNN_CIFAR(method, nn.Module):
    data = None

    max_epoch = 40       # 加深网络后,增加 epoch 让其充分收敛
    learning_rate = 1e-3
    batch_size = 64

    train_loss_list = []
    train_acc_list = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # input: (N,3,32,32)
        # Conv1: 3->32, k=3, p=1 -> (32,32,32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # -> (32,16,16)

        # Conv2: 32->64, k=3, p=1 -> (64,16,16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # -> (64,8,8)

        # Conv3: 64->128, k=3, p=1 -> (128,8,8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)   # -> (128,4,4)

        # Conv4: 128->256, k=3, p=1 -> (256,4,4)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)   # -> (256,2,2)

        self.flatten = nn.Flatten()
        # 256 * 2 * 2 = 1024
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.act5 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)      # 10 classes, output logits

    def forward(self, x):
        '''Forward propagation, output logits (no softmax)'''
        h = self.pool1(self.act1(self.bn1(self.conv1(x))))     # (32,16,16)
        h = self.pool2(self.act2(self.bn2(self.conv2(h))))     # (64,8,8)
        h = self.pool3(self.act3(self.bn3(self.conv3(h))))     # (128,4,4)
        h = self.pool4(self.act4(self.bn4(self.conv4(h))))     # (256,2,2)
        h = self.flatten(h)
        h = self.dropout(self.act5(self.fc1(h)))
        logits = self.fc2(h)
        return logits

    def train(self, X, y):
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        
        # 增加学习率衰减,帮助模型后期收敛更好
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 35], gamma=0.1)

        self.train_loss_list = []
        self.train_acc_list = []

        for epoch in range(self.max_epoch):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item() * X_batch.size(0)
                pred_labels = y_pred.max(1)[1]
                correct += (pred_labels == y_batch).sum().item()
                total += y_batch.size(0)

            avg_loss = epoch_loss / total
            acc = correct / total
            self.train_loss_list.append(avg_loss)
            self.train_acc_list.append(acc)
            
            scheduler.step()  # 更新学习率

            print('Epoch:', epoch, 'Loss:', round(avg_loss, 4), 'Accuracy:', round(acc, 4))

    def test(self, X):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.array(X))
            logits = self.forward(X_tensor)
            return logits.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}