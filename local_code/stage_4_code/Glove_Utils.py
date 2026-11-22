import numpy as np
import os

class GloveUtils:
    def __init__(self, glove_path, dim=50, max_vocab=80000):
        """
        Utility class to load GloVe embeddings and encode text to token IDs.
        :param glove_path: Path to the GloVe text file.
        :param dim: Dimension of the word vectors.
        :param max_vocab: Maximum number of words to load (prevents OOM and slow training).
        """
        self.dim = dim
        self.max_vocab = max_vocab
        
        # Index 0 reserved for <pad>, Index 1 reserved for <unk>
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        
        # Initialize vectors for <pad> and <unk>
        self.vectors = [
            np.zeros(dim, dtype=np.float32), 
            np.random.normal(0, 0.1, dim).astype(np.float32)
        ]
        
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe file not found: {glove_path}")
        
        print("Loading GloVe embeddings...")
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Stop loading when the vocabulary limit is reached
                if len(self.vectors) >= self.max_vocab:
                    break
                    
                parts = line.rstrip().split(' ')
                word = parts[0]
                
                if word in self.word2idx:
                    continue
                    
                vec = np.asarray(parts[1:], dtype=np.float32)
                self.word2idx[word] = len(self.vectors)
                self.idx2word[len(self.vectors)] = word
                self.vectors.append(vec)
                
        self.embedding_matrix = np.stack(self.vectors)
        print(f"GloVe loaded. Vocab size: {len(self.word2idx)} (Capped at {self.max_vocab})")

    def encode(self, tokens, max_len):
        """
        Convert a list of tokens to a fixed-length list of IDs.
        """
        ids = [self.word2idx.get(token, 1) for token in tokens]
        
        # Truncate
        if len(ids) > max_len:
            ids = ids[:max_len]
        # Pad
        elif len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
            
        return ids