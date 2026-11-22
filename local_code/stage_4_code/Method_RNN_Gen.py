import time
import torch
import torch.nn as nn
import numpy as np
from local_code.base_class.method import method

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class RNNGenerator(nn.Module):
    def __init__(self, emb_matrix, hidden_size=256, cell_type='rnn', dropout=0.3, num_layers=1):
        super(RNNGenerator, self).__init__()
        vocab_size, emb_dim = emb_matrix.shape
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        
        rnn_dropout = dropout if num_layers > 1 else 0.0

        if cell_type == 'rnn':
            self.rnn = nn.RNN(emb_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        elif cell_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        elif cell_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
            
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.rnn(emb, hidden)
        return self.fc(out), hidden

class Method_RNN_Gen(method):
    def __init__(self, mName="RNN_Generator", mDescription="RNN Text Generator", emb_matrix=None, glove=None, cell_type='rnn', hidden_size=256, epochs=20, batch_size=64, lr=1e-3, num_layers=1):
        self.method_name = mName
        self.method_description = mDescription
        self.data = None
        self.method_start_time = None
        self.method_stop_time = None
        self.method_running_time = None
        self.method_training_time = None
        self.method_testing_time = None
        
        self.model = RNNGenerator(emb_matrix, hidden_size, cell_type=cell_type, num_layers=num_layers).to(DEVICE)
        self.glove = glove
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.history = {'train_loss': []}

    def run(self, trainData, trainLabel, testData):
        self.method_start_time = time.time()
        X_train = torch.from_numpy(trainData).to(DEVICE)
        y_train = torch.from_numpy(trainLabel).to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        t0 = time.time()
        print(f"--- Training {self.method_description} ---")
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(X_train.size(0))
            X_train, y_train = X_train[perm], y_train[perm]
            
            total_loss = 0.0
            for i in range(0, len(X_train), self.batch_size):
                batch_x, batch_y = X_train[i:i+self.batch_size], y_train[i:i+self.batch_size]
                optimizer.zero_grad()
                logits, _ = self.model(batch_x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
                
            avg_loss = total_loss / len(X_train)
            self.history['train_loss'].append(avg_loss)
            print(f"Epoch [{epoch+1}/{self.epochs}] | Gen Loss: {avg_loss:.4f}")
            
        self.method_training_time = time.time() - t0
        
        generated_texts = []
        for prefix_words in testData:
            generated_texts.append(self._generate(prefix_words))
            
        self.method_stop_time = time.time()
        return generated_texts

    @torch.no_grad()
    def _generate(self, prefix_words, max_new_tokens=25, temperature=0.8):
        self.model.eval()
        prefix_ids = [self.glove.word2idx.get(w, 1) for w in prefix_words]
        seq = torch.tensor([prefix_ids], dtype=torch.long, device=DEVICE)
        
        for _ in range(max_new_tokens):
            logits, _ = self.model(seq)
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            seq = torch.cat([seq, torch.tensor([[next_token]], device=DEVICE)], dim=1)
            if self.glove.idx2word[next_token] in ['.', '?', '!']:
                break
                
        generated_tokens = [self.glove.idx2word[i] for i in seq[0].tolist()]
        text = ' '.join(generated_tokens)
        return text.replace(" ?", "?").replace(" .", ".").replace(" !", "!").replace(" ,", ",")