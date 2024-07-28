import csv

from utils import flop_estimator
from flop_estimators import classifier_flop_estimator, FS_flop_estimator, ODDS_flop_estimator, SHAP_flop_estimator, SID_flop_estimator, CCGAN_flop_estimator

flop_estimators = [
	classifier_flop_estimator(),
	FS_flop_estimator(),
	ODDS_flop_estimator(),
	SHAP_flop_estimator(),
    SID_flop_estimator(),
	CCGAN_flop_estimator()
]

for estimator in flop_estimators:
    estimator.mnist_params()
    estimator.cifar10_params()
    estimator.mnist_train_flops()
    estimator.mnist_test_flops()
    estimator.cifar10_train_flops()
    estimator.cifar10_test_flops()


data = flop_estimator._cache

def export_results(target_type, filename):
    csv_data = []

    def insert_flops(method, dataset, flops):
        col_index = 1 + int(dataset != 'mnist')
        try:
            row_index = [row[0] for row in csv_data].index(method)
            csv_data[row_index][col_index] = flops
        except ValueError:
            row = [method, -1, -1]
            row[col_index] = flops
            csv_data.append(row)

    for key, value in data.items():
        method, dataset, type = key.rsplit('_')
        if type == target_type:
            if isinstance(value, dict):
                for sub_method, flops in value.items():
                    insert_flops(f"{method}\\\\{sub_method}", dataset, flops)
            else:
                insert_flops(method, dataset, value)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Label', 'MNIST', 'CIFAR-10'])
        writer.writerows(csv_data)

export_results("test", "test_fpos.csv")
export_results("train", "train_fpos.csv")
export_results("params", "params.csv")
