import os
import subprocess

os.chdir("/")

for data_type in ['svhn', 'syn', 'usps', 'mnistm']:
    for extra_data_type in ['mnist', 'svhn', 'syn', 'usps', 'mnistm']:
        print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: {data_type} {extra_data_type} -----")
        subprocess.run(
            ["python",
             "CBA/main.py",
             "--data_type", data_type,
             "--extra_data_type", extra_data_type,
             "--home_path", f"{data_type}+"])
