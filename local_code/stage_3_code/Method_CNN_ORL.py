from local_code.base_class.method import method
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class Method_CNN_ORL(method, nn.Module):
    data = None

    max_epoch = 15
    learning_rate = 1e-3
    batch_size = 16 

    train_loss_list = []
    train_acc_list = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # input: (N,1,112,92)
        # Conv1: 3->16, k=3, p=1 -> (16,112,92)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        # pool1: 112x92 -> 56x46
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv2: 16->32, k=3, p=1 -> (32,56,46)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        # pool2: 56x46 -> 28x23
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv3: 32->64, k=3, p=1 -> (64,28,23)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        # pool3: 28x23 -> 14x11
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        # 64 * 14 * 11 = 9856
        self.fc1 = nn.Linear(64 * 14 * 11, 128)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 40)      # 40 classes, output logits

    def forward(self, x):
        '''Forward propagation, output logits (no softmax)'''
        h = self.pool1(self.act1(self.conv1(x)))     # (16,56,46)
        h = self.pool2(self.act2(self.conv2(h)))     # (32,28,23)
        h = self.pool3(self.act3(self.conv3(h)))     # (64,14,11)
        h = self.flatten(h)
        h = self.act4(self.fc1(h))
        logits = self.fc2(h)
        return logits

    def train(self, X, y):
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

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