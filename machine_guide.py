import winreg

def get_machine_guid():
    key = winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"SOFTWARE\Microsoft\Cryptography"
    )
    value, _ = winreg.QueryValueEx(key, "MachineGuid")
    return value

print(get_machine_guid())

def serial_autorizado():
    return "76c72170-542d-4d9b-a584-f47af2a7f3c4"
    