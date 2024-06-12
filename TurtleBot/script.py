import subprocess

for tipo_ben in ['1', '2', 'der', 'izq']:
    for tipo_mal in ['1', '2', 'der', 'izq']:
        print(f"----- [NUEVA EJECUCIÓN] turtlebot.py con argumentos: {tipo_ben} {tipo_mal} -----")
        subprocess.run(
            ["python",
                "main.py",
                "--home_path", "ScriptDef",
                "--tipo_ben", tipo_ben,
                "--tipo_mal", tipo_mal])


"""
for combinacion in [('1', 'der'), ('2', 'izq')]:
    tipo_ben, tipo_mal = combinacion
    print(f"----- [NUEVA EJECUCIÓN] turtlebot.py con argumentos: {tipo_ben} {tipo_mal} -----")
    subprocess.run(
        ["python",
         "main.py",
         "--home_path", "TurtleBotScriptRepetidos",
         "--tipo_ben", tipo_ben,
         "--tipo_mal", tipo_mal])
"""