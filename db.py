import mysql.connector

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="sve",          
        password="sve123#",     
        database="reconhecimento_facial"
    )



