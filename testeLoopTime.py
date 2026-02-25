
from datetime import datetime
import threading
import time

global fim_execucao
# Função que será executada em uma thread
def tarefa_em_loop():
    while True:
        inicio = datetime.now()  # Início da medição
        if(fim_execucao-inicio) > 3600: # Se passou mais de 1 hora
            print("Passou mais de 1 hora")
            break

            
            
         

# Criando e iniciando a thread
thread = threading.Thread(target=tarefa_em_loop)
thread.daemon = True  # Encerra a thread quando o programa principal terminar
thread.start()