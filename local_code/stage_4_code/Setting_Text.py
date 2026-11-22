from local_code.base_class.setting import setting

class Setting_Text(setting):
    def __init__(self, sName="Setting", sDescription="Setting Description"):
        self.setting_name = sName
        self.setting_description = sDescription
        self.dataset = None
        self.method = None
        self.result = None
        self.evaluate = None

    def load_run_save_evaluate(self):
        data = self.dataset.load()
        
        if "Classification" in self.dataset.dataset_name:
            preds = self.method.run(data['train_X'], data['train_y'], data['test_X'])
            metrics = self.evaluate.evaluate(preds, data['test_y'])
        
        elif "Generation" in self.dataset.dataset_name:
            prefixes = [
                ["what", "did", "the"],
                ["why", "did", "the"],
                ["i", "like", "my"]
            ]
            generated_texts = self.method.run(data['train_X'], data['train_y'], prefixes)
            metrics = self.evaluate.evaluate(generated_texts)
            
            print("\n--- Generated Jokes ---")
            for prefix, text in zip(prefixes, generated_texts):
                print(f"Start: {' '.join(prefix)} -> {text}")
            print("-----------------------\n")

        self.result.save(metrics=metrics, history=self.method.history)
        return metrics