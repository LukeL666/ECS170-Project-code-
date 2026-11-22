import time
import torch
import torch.nn as nn
import numpy as np
from local_code.base_class.method import method

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class RNNClassifier(nn.Module):
    def __init__(self, emb_matrix, hidden_size=128, num_classes=2, cell_type='rnn', dropout=0.6):
        super(RNNClassifier, self).__init__()
        vocab_size, emb_dim = emb_matrix.shape
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        
        if cell_type == 'rnn':
            self.rnn = nn.RNN(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        elif cell_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        elif cell_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hidden_size, batch_first=True, bidirectional=True)
            
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        mask = (x != 0).float() 
        emb = self.embedding(x) 
        out, _ = self.rnn(emb)  
        out = out * mask.unsqueeze(-1)
        pooled = out.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return self.fc(self.dropout(pooled))

class Method_RNN_Cls(method):
    def __init__(self, mName="RNN_Classifier", mDescription="RNN Text Classifier", emb_matrix=None, cell_type='rnn', hidden_size=128, epochs=10, batch_size=64, lr=1e-3):
        # Manual assignment
        self.method_name = mName
        self.method_description = mDescription
        self.data = None
        self.method_start_time = None
        self.method_stop_time = None
        self.method_running_time = None
        self.method_training_time = None
        self.method_testing_time = None
        
        self.model = RNNClassifier(emb_matrix, hidden_size, num_classes=2, cell_type=cell_type).to(DEVICE)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    def run(self, trainData, trainLabel, testData):
        self.method_start_time = time.time()
        
        # Split validation set
        val_size = int(len(trainData) * 0.1)
        val_X, val_y = trainData[:val_size], trainLabel[:val_size]
        X_train, y_train = trainData[val_size:], trainLabel[val_size:]
        
        X_train = torch.from_numpy(X_train).to(DEVICE)
        y_train = torch.from_numpy(y_train).to(DEVICE)
        val_X = torch.from_numpy(val_X).to(DEVICE)
        val_y = torch.from_numpy(val_y).to(DEVICE)
        X_test = torch.from_numpy(testData).to(DEVICE)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        print(f"--- Training {self.method_description} on {DEVICE} ---")
        t0 = time.time()
        
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(X_train.size(0))
            X_train, y_train = X_train[perm], y_train[perm]
            
            total_loss, correct = 0.0, 0
            for i in range(0, len(X_train), self.batch_size):
                batch_x = X_train[i:i+self.batch_size]
                batch_y = y_train[i:i+self.batch_size]
                
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                
                total_loss += loss.item() * len(batch_x)
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
                
            train_loss = total_loss / len(X_train)
            train_acc = correct / len(X_train)
            
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_X)
                val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()
                
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch [{epoch+1}/{self.epochs}] | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        self.method_training_time = time.time() - t0
        
        self.model.eval()
        test_preds = []
        with torch.no_grad():
            # Use a batch loop for testing to prevent CUDA OOM
            for i in range(0, len(X_test), self.batch_size):
                batch_x = X_test[i:i+self.batch_size]
                logits = self.model(batch_x)
                preds = logits.argmax(dim=1).cpu().numpy()
                test_preds.extend(preds)
        
        test_preds = np.array(test_preds)
            
        self.method_stop_time = time.time()
        self.method_running_time = self.method_stop_time - self.method_start_time
        return test_preds