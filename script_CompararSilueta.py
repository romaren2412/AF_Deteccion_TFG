import subprocess
import datetime

# attacks = ["backdoor", "dba", "full_mean_attack", "full_trim", "partial_trim"]
attacks = ["backdoor", "dba", "full_mean_attack", "full_trim"]
aggregations = ['median']
aggregations2 = ['simple_mean', 'trim', 'median']

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

"""
SILUETA 90
full mean attack --> simple_mean
full_trim --> median, trim
"""

# MEDIAN FULL_TRIM CON SILUETA 00
print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: MEDIAN, FULL_TRIM, SIL90 -----")
subprocess.run(
    ["python",
     "mnist_CNN.py",
     "--byz_type", "full_trim",
     "--aggregation", "median",
     "--home_path", "ProbaSiluetav5_novo/Silueta_90/",
     "--timestamp", timestamp,
     "--tipo_exec", "loop",
     "--silhouette", str(0.9),
     "--nepochs", str(200)])

# TRIM FULL_TRIM CON SILUETA 00
print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: TRIM, FULL_TRIM, SIL90 -----")
subprocess.run(
    ["python",
     "mnist_CNN.py",
     "--byz_type", "full_trim",
     "--aggregation", "trim",
     "--home_path", "ProbaSiluetav5_novo/Silueta_90/",
     "--timestamp", timestamp,
     "--tipo_exec", "loop",
     "--silhouette", str(0.9),
     "--nepochs", str(200)])


"""
SILUETA 70
full_trim --> trim
"""

# TRIM FULL_TRIM CON SILUETA 70
print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: TRIM, FULL_TRIM, SIL70 -----")
subprocess.run(
    ["python",
     "mnist_CNN.py",
     "--byz_type", "full_trim",
     "--aggregation", "trim",
     "--home_path", "ProbaSiluetav5_novo/Silueta_70/",
     "--timestamp", timestamp,
     "--tipo_exec", "loop",
     "--silhouette", str(0.7),
     "--nepochs", str(200)])

"""
SILUETA 00
backdoor --> median
full_mean --> median
full_trim --> median, trim
"""

# MEDIAN BACKDOOR CON SILUETA 00
print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: MEDIAN, BACKDOOR, SIL00 -----")
subprocess.run(
    ["python",
     "mnist_CNN.py",
     "--byz_type", "backdoor",
     "--aggregation", "median",
     "--home_path", "ProbaSiluetav5_novo/Silueta_00/",
     "--timestamp", timestamp,
     "--tipo_exec", "loop",
     "--silhouette", str(0.0),
     "--nepochs", str(200)])

# MEDIAN BACKDOOR CON SILUETA 00
print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: MEDIAN, FULL_MEAN, SIL00 -----")
subprocess.run(
    ["python",
     "mnist_CNN.py",
     "--byz_type", "full_mean_attack",
     "--aggregation", "median",
     "--home_path", "ProbaSiluetav5_novo/Silueta_00/",
     "--timestamp", timestamp,
     "--tipo_exec", "loop",
     "--silhouette", str(0.0),
     "--nepochs", str(200)])

# MEDIAN FULL_TRIM CON SILUETA 00
print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: MEDIAN, FULL_TRIM, SIL00 -----")
subprocess.run(
    ["python",
     "mnist_CNN.py",
     "--byz_type", "full_trim",
     "--aggregation", "median",
     "--home_path", "ProbaSiluetav5_novo/Silueta_00/",
     "--timestamp", timestamp,
     "--tipo_exec", "loop",
     "--silhouette", str(0.0),
     "--nepochs", str(200)])

# TRIM FULL_TRIM CON SILUETA 00
print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: TRIM, FULL_TRIM, SIL00 -----")
subprocess.run(
    ["python",
     "mnist_CNN.py",
     "--byz_type", "full_trim",
     "--aggregation", "trim",
     "--home_path", "ProbaSiluetav5_novo/Silueta_00/",
     "--timestamp", timestamp,
     "--tipo_exec", "loop",
     "--silhouette", str(0.0),
     "--nepochs", str(200)])
