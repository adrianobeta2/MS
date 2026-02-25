def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as file:
            temp = int(file.read()) / 1000.0  # O valor vem em milicelsius
            return temp
    except FileNotFoundError:
        return None  # Retorna None se o arquivo não for encontrado

temperature = get_cpu_temperature()
temperatura = f"{temperature:.2f}°C"
if temperature is not None:
    print(f"Temperatura da CPU: {temperature:.2f}°C")
else:
    print("Não foi possível ler a temperatura.")
