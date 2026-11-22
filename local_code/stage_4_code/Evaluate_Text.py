import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from local_code.base_class.evaluate import evaluate

class Evaluate_TextCls(evaluate):
    def __init__(self, eName="TextCls_Evaluate", eDescription="Classification Metrics"):
        self.evaluate_name = eName
        self.evaluate_description = eDescription
        self.data = None

    def evaluate(self, preds, labels):
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        return {'accuracy': float(acc), 'precision': float(p), 'recall': float(r), 'f1': float(f1)}

class Evaluate_TextGen(evaluate):
    def __init__(self, eName="TextGen_Evaluate", eDescription="Generation Metrics"):
        self.evaluate_name = eName
        self.evaluate_description = eDescription
        self.data = None

    def evaluate(self, generated_texts):
        uniq_ratios = [len(set(text.split())) / max(len(text.split()), 1) for text in generated_texts]
        return {'avg_unique_word_ratio': float(np.mean(uniq_ratios)), 'samples': generated_texts}