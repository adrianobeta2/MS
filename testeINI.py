import configparser

# Carregar o arquivo INI
config = configparser.ConfigParser()
config.read('config.ini')

# Obter valores booleanos
cor = config.getboolean('Ferramentas', 'cor')
textura = config.getboolean('Ferramentas', 'textura')
pixel = config.getboolean('Ferramentas', 'pixel')

# Verificar os valores
print(cor)     # True
print(textura) # False
print(pixel)   # True
