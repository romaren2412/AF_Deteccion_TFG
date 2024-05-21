import subprocess
import os

os.chdir("../")

for data_type in ["mnist", "mnistm", "svhn", "syn", "usps"]:
    for extra_data_type in ["mnist", "mnistm", "svhn", "syn", "usps"]:
        if data_type == extra_data_type:
            print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: {data_type} {extra_data_type} -----")
            subprocess.run(
                ["python",
                    "DigitFive/D2_CatroBases_Adicional/main.py",
                    "--data_type", data_type,
                    "--extra_data_type", extra_data_type,
                    "--tipo_exec", "no_detect",
                    "--home_path", f"Iguais/{data_type}_{extra_data_type}"])
