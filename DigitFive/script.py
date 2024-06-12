import subprocess
import os

# opt = 'Minibatches'
opt = '1Batch'

for data_type in ['mnist', 'svhn', 'syn', 'usps', 'mnistm']:
    for extra_data_type in ['mnist', 'svhn', 'syn', 'usps', 'mnistm']:
        if data_type == extra_data_type:
            print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: {data_type} {extra_data_type} -----")
            subprocess.run(
                ["python",
                    f"{opt}/main.py",
                    "--data_type", data_type,
                    "--extra_data_type", extra_data_type,
                    "--home_path", f"ProbasSCRIPT/{data_type}/{data_type}_{extra_data_type}"])
