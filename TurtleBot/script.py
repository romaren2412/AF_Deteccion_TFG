import subprocess

for tipo_ben in ['1', '2', 'der', 'izq']:
    for tipo_mal in ['1', '2', 'der', 'izq']:
        print(f"----- [NUEVA EJECUCIÓN] turtlebot.py con argumentos: {tipo_ben} {tipo_mal} -----")
        subprocess.run(
            ["python",
                "main.py",
                "--home_path", "TurtleBotScript_20epochs",
                "--tipo_ben", tipo_ben,
                "--tipo_mal", tipo_mal])
