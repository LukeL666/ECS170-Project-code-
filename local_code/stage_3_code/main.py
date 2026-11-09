# At ECS170_Fall_2026_Source_Code_Template directory， run the following command to start the program:
# python -m local_code.stage_3_code.main


import os
from matplotlib import pyplot as plt

from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
from local_code.stage_3_code.Dataset_Loader_ORL import Dataset_Loader_ORL
from local_code.stage_3_code.Dataset_Loader_CIFAR import Dataset_Loader_CIFAR

from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR

from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_1_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics

DATASET = ['CIFAR']

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
DATA_FOLDER = os.path.join(project_root, 'data', 'stage_3_data')


def build_components(dataset_name):
    if dataset_name == 'MNIST':
        ds = Dataset_Loader_MNIST('MNIST', 'hand-written digits')
        ds.dataset_source_folder_path = DATA_FOLDER
        ds.dataset_source_file_name = 'MNIST'
        method = Method_CNN_MNIST('CNN-MNIST', 'CNN for MNIST')
        result_prefix = 'result_mnist'
    elif dataset_name == 'ORL':
        ds = Dataset_Loader_ORL('ORL', 'face images')
        ds.dataset_source_folder_path = DATA_FOLDER
        ds.dataset_source_file_name = 'ORL'
        method = Method_CNN_ORL('CNN-ORL', 'CNN for ORL')
        result_prefix = 'result_orl'
    elif dataset_name == 'CIFAR':
        ds = Dataset_Loader_CIFAR('CIFAR', 'colored objects')
        ds.dataset_source_folder_path = DATA_FOLDER
        ds.dataset_source_file_name = 'CIFAR'
        method = Method_CNN_CIFAR('CNN-CIFAR', 'CNN for CIFAR')
        result_prefix = 'result_cifar'
    else:
        raise ValueError('unknown dataset: ' + dataset_name)

    result = Result_Saver('saver', '')
    result.result_destination_folder_path = './result/'
    result.result_destination_file_name = result_prefix
    result.fold_count = 0

    evaluate = Evaluate_Metrics('metrics', '')

    setting = Setting_Train_Test_Split('train-test split', dataset_name)
    setting.prepare(ds, method, result, evaluate)
    return setting, method


def plot_learning_curve(method, dataset_name):
    plt.figure()
    plt.plot(method.train_loss_list, label='train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(dataset_name + ' training loss')
    plt.legend()
    plt.savefig('./result/' + dataset_name + '_loss.png')
    plt.show()

    plt.figure()
    plt.plot(method.train_acc_list, label='train F1 score')
    plt.xlabel('epoch')
    plt.ylabel('F1 score')
    plt.title(dataset_name + ' training F1 score')
    plt.legend()
    plt.savefig('./result/' + dataset_name + '_acc.png')
    plt.show()


if __name__ == '__main__':
    os.makedirs('./result/', exist_ok=True)

    for data in DATASET:
        setting, method = build_components(data)

        print(f'************ {data} start ************')
        setting.print_setup_summary()

        score, _ = setting.load_run_save_evaluate()
        print('Test accuracy:', score)

        plot_learning_curve(method, data)

        print(f'************ {data} end ************')