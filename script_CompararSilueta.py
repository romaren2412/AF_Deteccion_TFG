import subprocess
import datetime

# attacks = ["backdoor", "dba", "full_mean_attack", "full_trim", "partial_trim"]
attacks = ["backdoor", "dba", "full_mean_attack", "full_trim"]
aggregations = ['median']
aggregations2 = ['simple_mean', 'trim', 'median']

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Bucle para ejecutar main.py con cada argumento
for agr in aggregations:
    for attack in attacks:
        print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: {agr}, {attack} -----")
        subprocess.run(
            ["python",
             "mnist_CNN.py",
             "--byz_type", attack,
             "--aggregation", agr,
             "--home_path", "ProbaSiluetav2/Silueta_00/",
             "--timestamp", timestamp,
             "--tipo_exec", "loop",
             "--silhouette", str(0.0),
             "--nepochs", str(200)])

# TRIM FULL_TRIM CON SILUETA 00
print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: TRIM, FULLTRIM, SIL00 -----")
subprocess.run(
    ["python",
     "mnist_CNN.py",
     "--byz_type", "full_trim",
     "--aggregation", "trim",
     "--home_path", "ProbaSiluetav2/Silueta_00/",
     "--timestamp", timestamp,
     "--tipo_exec", "loop",
     "--silhouette", str(0.0),
     "--nepochs", str(200)])

# TRIM FULL_TRIM CON SILUETA
print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: TRIM, FULLTRIM, SIL70 -----")
subprocess.run(
    ["python",
     "mnist_CNN.py",
     "--byz_type", "full_trim",
     "--aggregation", "trim",
     "--home_path", "ProbaSiluetav2/Silueta_70/",
     "--timestamp", timestamp,
     "--tipo_exec", "loop",
     "--silhouette", str(0.7),
     "--nepochs", str(200)])


# TODAS CON SILUETA 90
# Bucle para ejecutar main.py con cada argumento
for agr in aggregations2:
    for attack in attacks:
        print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: {agr}, {attack} -----")
        subprocess.run(
            ["python",
             "mnist_CNN.py",
             "--byz_type", attack,
             "--aggregation", agr,
             "--home_path", "ProbaSiluetav2/Silueta_90/",
             "--timestamp", timestamp,
             "--tipo_exec", "loop",
             "--silhouette", str(0.9),
             "--nepochs", str(200)])
