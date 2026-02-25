import os

def get_serial():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Serial"):
                    return line.strip().split(":")[1].strip()
    except Exception as e:
        return None

if __name__ == "__main__":
    serial_autorizado = "10000000abcd1234"  # Defina o serial autorizado
    serial_atual = get_serial()

    if serial_atual == serial_autorizado:
        print("Dispositivo autorizado. Executando o código...")
        # Coloque seu código principal aqui
    else:
        print("Acesso negado! Dispositivo não autorizado.")
        exit(1)
