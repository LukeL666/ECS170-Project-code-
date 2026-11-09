from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

class Evaluate_Metrics(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        if isinstance(true_y, torch.Tensor):
            true_y = true_y.numpy()
        if isinstance(pred_y, torch.Tensor):
            pred_y = pred_y.numpy()

        acc = accuracy_score(true_y, pred_y)
        precision = precision_score(true_y, pred_y, average='weighted', zero_division=0)
        recall = recall_score(true_y, pred_y, average='weighted', zero_division=0)
        f1 = f1_score(true_y, pred_y, average='weighted', zero_division=0)
        
        print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")
        
        return f1