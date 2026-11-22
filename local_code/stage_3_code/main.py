# At ECS170_Fall_2026_Source_Code_Template directory, run the following command to start the program:
# python -m local_code.stage_4_code.main

import os
from matplotlib import pyplot as plt

from local_code.stage_4_code.Glove_Utils import GloveUtils
from local_code.stage_4_code.Dataset_Loader_TextCls import Dataset_Loader_TextCls
from local_code.stage_4_code.Dataset_Loader_TextGen import Dataset_Loader_TextGen
from local_code.stage_4_code.Method_RNN_Cls import Method_RNN_Cls
from local_code.stage_4_code.Method_RNN_Gen import Method_RNN_Gen
from local_code.stage_4_code.Evaluate_Text import Evaluate_TextCls, Evaluate_TextGen
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Text import Setting_Text

# Resolve project root directory dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
DATA_FOLDER = os.path.join(project_root, 'data')

# Initialize global GloVe utility
print("Initializing GloVe Utils (this may take a minute)...")
GLOVE_PATH = os.path.join(DATA_FOLDER, "Glove", "wiki_giga_2024_200_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined")
glove = GloveUtils(GLOVE_PATH, dim=200)


def build_cls_components(cell_type):
    ds = Dataset_Loader_TextCls(data_root_path=os.path.join(DATA_FOLDER, 'text_classification'), glove_util=glove, max_len=200)
    method = Method_RNN_Cls(mName=f"RNN_Cls_{cell_type}", mDescription=f"{cell_type.upper()} Cls", emb_matrix=glove.embedding_matrix, cell_type=cell_type, hidden_size=128, epochs=5, batch_size=64)
    result = Result_Saver(rName=f"cls_result_{cell_type}", folder_path='./results/')
    evaluate = Evaluate_TextCls()
    setting = Setting_Text(sName=f"Setting_Cls_{cell_type}", sDescription=f"Cls {cell_type}")
    setting.prepare(ds, method, result, evaluate)
    return setting, method


def build_gen_components(cell_type):
    ds = Dataset_Loader_TextGen(file_path=os.path.join(DATA_FOLDER, 'text_generation', 'data'), glove_util=glove, max_len=30)
    method = Method_RNN_Gen(mName=f"RNN_Gen_{cell_type}", mDescription=f"{cell_type.upper()} Gen", emb_matrix=glove.embedding_matrix, glove=glove, cell_type=cell_type, hidden_size=256, epochs=30, batch_size=32)
    result = Result_Saver(rName=f"gen_result_{cell_type}", folder_path='./results/')
    evaluate = Evaluate_TextGen()
    setting = Setting_Text(sName=f"Setting_Gen_{cell_type}", sDescription=f"Gen {cell_type}")
    setting.prepare(ds, method, result, evaluate)
    return setting, method


def plot_learning_curve_cls(method, cell_type):
    """Extract metrics from method.history and save learning curves."""
    history = method.history
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Training Loss
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(cell_type.upper() + ' training loss')
    plt.legend()
    plt.savefig(f'./results/cls_{cell_type}_loss.png')
    plt.close() 
    
    # Plot Train and Validation Accuracy
    plt.figure()
    plt.plot(epochs, history['train_acc'], label='train acc')
    plt.plot(epochs, history['val_acc'], label='val acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title(cell_type.upper() + ' accuracy')
    plt.legend()
    plt.savefig(f'./results/cls_{cell_type}_acc.png')
    plt.close()


if __name__ == '__main__':
    os.makedirs('./results/', exist_ok=True)
    CELLS = ['rnn', 'lstm', 'gru']

    # --- Part 1: Text Classification ---
    print('\n' + '='*40)
    print('STARTING TEXT CLASSIFICATION EXPERIMENTS')
    print('='*40)
    
    for cell in CELLS:
        setting, method = build_cls_components(cell)
        
        print(f'************ Classification ({cell.upper()}) start ************')
        setting.print_setup_summary()
        
        setting.load_run_save_evaluate()
        plot_learning_curve_cls(method, cell)
        
        print(f'************ Classification ({cell.upper()}) end ************\n')

    # --- Part 2: Text Generation ---
    print('\n' + '='*40)
    print('STARTING TEXT GENERATION EXPERIMENTS')
    print('='*40)
    
    for cell in CELLS:
        setting, method = build_gen_components(cell)
        
        print(f'************ Generation ({cell.upper()}) start ************')
        setting.print_setup_summary()
        
        setting.load_run_save_evaluate()
        print(f'************ Generation ({cell.upper()}) end ************\n')

    print("All experiments completed! Check the /results folder.")