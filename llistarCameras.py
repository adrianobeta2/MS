import subprocess

def listar_cameras():
    try:
        resultado = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True
        )
        print(resultado.stdout)
    except FileNotFoundError:
        print("O comando v4l2-ctl não foi encontrado. Certifique-se de que o pacote v4l-utils está instalado.")

listar_cameras()
