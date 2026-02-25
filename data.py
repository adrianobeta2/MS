
from datetime import datetime

aprovados = 0
rejeitados = 0

dia= 0
dia_anterior = 0




def status():
  agora = datetime.now()

  dia = agora.day
 
  if(dia != dia_anterior){
    aprovados = 0
  }
  else:
    {
      aprovados +=1

    }
      
  print(agora)

