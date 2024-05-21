import subprocess

# same for 2
print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: BACK DET -----")
subprocess.run(
    ["python",
        "main.py",
        "--home_path", "D",
        "--byz_type", "backdoor",
        "--tipo_exec", "detect"])

print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: BACK NO DET -----")
subprocess.run(
    ["python",
        "main.py",
        "--home_path", "ND",
        "--byz_type", "backdoor",
        "--tipo_exec", "no_detect"])

print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: MEAN DET -----")
subprocess.run(
    ["python",
        "main.py",
        "--home_path", "D",
        "--byz_type", "mean_attack",
        "--tipo_exec", "detect"])

print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: MEAN NO DET -----")
subprocess.run(
    ["python",
        "main.py",
        "--home_path", "ND",
        "--byz_type", "mean_attack",
        "--tipo_exec", "no_detect"])

print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: BSP DET -----")
subprocess.run(
    ["python",
        "main.py",
        "--home_path", "D",
        "--byz_type", "backdoor_sen_pixel",
        "--tipo_exec", "detect"])

print(f"----- [NUEVA EJECUCIÓN] mnist_CNN.py con argumentos: BSP NO DET -----")
subprocess.run(
    ["python",
        "main.py",
        "--home_path", "ND",
        "--byz_type", "backdoor_sen_pixel",
        "--tipo_exec", "no_detect"])