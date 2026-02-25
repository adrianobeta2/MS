from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle
from kivy.config import Config
from threading import Thread
import requests
import configparser
from kivy.clock import Clock
from kivy.uix.image import Image
import os
from datetime import datetime

class ImageInicializacao(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar redimensionamento automático
        self.size = (300, 50)  # Definir tamanho fixo para a imagem
        self.pos_hint = {'center_x': 0.5, 'center_y': 0.5}  # Centraliza horizontal e verticalmente

        # Caminho da imagem que será exibida
        self.source = 'caminho/para/imagem.jpg'  # Substitua pelo caminho da sua imagem

    def set_image(self, image_path):
        # Atualiza a imagem exibida
        self.source = image_path




###########################################################################  REGIAO DE TELA #################################################################################

Config.set('graphics', 'fullscreen', 'auto')
class StatusGeral(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (220, 150)  # Definir largura e altura fixas
        self.pos = (10, 900)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 100  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        #self.text = "Conectando ao servidor..."
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0, 0.5, 0, 1)  # Cor inicial do fundo (verde escuro)
            self.rect = Rectangle(pos=self.pos, size=self.size)  # Retângulo de fundo

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def set_status_color(self, status):
        """Atualiza a cor de fundo e o texto conforme o status."""
        if status == "OK":
            self.bg_color.rgba = (0, 1, 0, 1)  # Verde
            self.text = "OK"
        elif status == "NOT_OK":
            self.bg_color.rgba = (1, 0, 0, 1)  # Vermelho
            self.text = "NOK"
        else:
            self.bg_color.rgba = (0.5, 0.5, 0.5, 1)  # Cinza
            self.text = "Status: Desconhecido"

class StatusParcial1(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (80, 60)  # Definir largura e altura fixas
        self.pos = (10, 700)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 40  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0, 0.5, 0, 1)  # Cor inicial do fundo (verde escuro)
            self.rect = Rectangle(pos=self.pos, size=self.size)  # Retângulo de fundo

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def set_status_color(self, status):
        """Atualiza a cor de fundo e o texto conforme o status."""
        if status == "OK":
            self.bg_color.rgba = (0, 1, 0, 1)  # Verde
            self.text = "OK"
        elif status == "NOT_OK":
            self.bg_color.rgba = (1, 0, 0, 1)  # Vermelho
            self.text = "NOK"
        else:
            self.bg_color.rgba = (0.5, 0.5, 0.5, 1)  # Cinza
            self.text = "Status: Desconhecido"

class LabelParcial1(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (200, 700)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "ROI 1"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class StatusParcial2(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (80, 60)  # Definir largura e altura fixas
        self.pos = (10, 600)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 40  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0, 0.5, 0, 1)  # Cor inicial do fundo (verde escuro)
            self.rect = Rectangle(pos=self.pos, size=self.size)  # Retângulo de fundo

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def set_status_color(self, status):
        """Atualiza a cor de fundo e o texto conforme o status."""
        if status == "OK":
            self.bg_color.rgba = (0, 1, 0, 1)  # Verde
            self.text = "OK"
        elif status == "NOT_OK":
            self.bg_color.rgba = (1, 0, 0, 1)  # Vermelho
            self.text = "NOK"
        else:
            self.bg_color.rgba = (0.5, 0.5, 0.5, 1)  # Cinza
            self.text = "Status: Desconhecido"
class LabelParcial2(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (200, 600)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "ROI 2"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class StatusParcial3(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (80, 60)  # Definir largura e altura fixas
        self.pos = (10, 500)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 40  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0, 0.5, 0, 1)  # Cor inicial do fundo (verde escuro)
            self.rect = Rectangle(pos=self.pos, size=self.size)  # Retângulo de fundo

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def set_status_color(self, status):
        """Atualiza a cor de fundo e o texto conforme o status."""
        if status == "OK":
            self.bg_color.rgba = (0, 1, 0, 1)  # Verde
            self.text = "OK"
        elif status == "NOT_OK":
            self.bg_color.rgba = (1, 0, 0, 1)  # Vermelho
            self.text = "NOK"
        else:
            self.bg_color.rgba = (0.5, 0.5, 0.5, 1)  # Cinza
            self.text = "Status: Desconhecido"

class LabelParcial3(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (200, 500)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "ROI 3"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class LabelAprovados(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (10, 300)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 26 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (0, 1, 0, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "PASS:"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class Aprovados(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (80, 300)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 26 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (0, 1, 0, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = ""
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class LabelRejeitados(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (5, 250)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 26 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 0, 0, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "FAIL:"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class Rejeitados(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (80, 250)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 26 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 0, 0, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = ""
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status


class LabelYield(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (10, 200)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 18 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (0, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "Yield(%):"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class Yield(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (80, 200)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 18 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (0, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = ""
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status
    
    

class LabelInicializacao(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar redimensionamento automático
        self.size = (300, 50)  # Definir tamanho fixo para a Label
        self.pos_hint = {'center_x': 0.5, 'center_y': 0.5}  # Centraliza horizontal e verticalmente

        self.font_size = 15  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = ""
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class ImageExibir(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar redimensionamento automático
        self.size = (1024, 768)  # Definir tamanho fixo para a imagem
        self.pos_hint = {'center_x': 0.5, 'center_y': 0.5}  # Centraliza horizontal e verticalmente
        self.allow_stretch = True  # Permite que a imagem se ajuste ao tamanho definido
        self.keep_ratio = True  # Mantém a proporção original da imagem

        # Caminho da imagem que será exibida
        self.source = 'imagem_status.jpg'  # Substitua pelo caminho da sua imagem

    def set_image(self, image_path):
        # Verifica se o caminho da imagem existe
        if os.path.exists(image_path):
            
                self.source = image_path  # Atualiza o caminho
                self.reload()  # Força o recarregamento da imagem
                self.canvas.ask_update()  # Solicita atualização no canvas
        else:
            print(f"Imagem não encontrada: {image_path}")
        

n_aprovados = 0
n_rejeitados = 0

dia = 0
dia_anterior = 0       
        

class MonitorApp(App):
    def build(self):
        
        layout = FloatLayout()  # Permite posicionamento absolutose
        self.imagem = ImageExibir()
        self.status_geral = StatusGeral()
        self.status_p1 = StatusParcial1()
        self.label_p1 = LabelParcial1()
        self.status_incial = LabelInicializacao()

        self.status_p2 = StatusParcial2()
        self.label_p2 = LabelParcial2()

        self.status_p3 = StatusParcial3()
        self.label_p3 = LabelParcial3()
       
        self.label_aprovados = LabelAprovados()
        self.aprovados = Aprovados()
        
        self.label_rejeitados = LabelRejeitados()
        self.rejeitados = Rejeitados()

        self.label_yield = LabelYield()
        self.yield_p = Yield()

        layout.add_widget(self.status_incial)
        layout.add_widget(self.status_geral)

        layout.add_widget(self.status_p1)
        layout.add_widget(self.label_p1)

        layout.add_widget(self.status_p2)
        layout.add_widget(self.label_p2)

        layout.add_widget(self.status_p3)
        layout.add_widget(self.label_p3)

        layout.add_widget(self.imagem)

        layout.add_widget(self.label_aprovados)
        layout.add_widget(self.aprovados)

        layout.add_widget(self.label_rejeitados)
        layout.add_widget(self.rejeitados)

        layout.add_widget(self.label_yield)
        layout.add_widget(self.yield_p)

        # Atualizar o status a cada 1 segundos
        Clock.schedule_interval(self.update_status, 1)

        return layout

    def update_status(self, dt):
        try:
            agora = datetime.now()
            global n_aprovados
            global n_rejeitados
            global dia
            global dia_anterior
            dia = agora.day
           
            # Ler o arquivo de configuração INI
            config = configparser.ConfigParser()
            config.read('status.ini')

            # Obter os parâmetros de ajuste de imagem
            executado    = str(config['STATUS']['executado'])
            resultado_p1 = str(config['STATUS']['parcial_p1'])
            resultado_p2 = str(config['STATUS']['parcial_p2'])
            resultado_p3 = str(config['STATUS']['parcial_p3'])
            resultado_geral = str(config['STATUS']['status_geral'])

            if(executado== "true"):
               # Aqui, tente imprimir para verificar se o caminho está correto
               print(f"Atualizando imagem com o caminho: imagem_status.jpg")
               Clock.schedule_once(lambda dt: self.imagem.set_image('imagem_status.jpg'))
               if(resultado_p1 =="True"):
                 self.status_p1.set_status_color("OK") 
               else:
                 self.status_p1.set_status_color("NOT_OK") 
                
               if(resultado_p2 =="True"):
                  self.status_p2.set_status_color("OK") 
               else:
                  self.status_p2.set_status_color("NOT_OK") 
                
               if(resultado_p3 =="True"):
                  self.status_p3.set_status_color("OK") 
               else:
                  self.status_p3.set_status_color("NOT_OK") 

               if(resultado_geral =="True"):
                  self.status_geral.set_status_color("OK") 
               else:
                  self.status_geral.set_status_color("NOT_OK") 
            
            
               if(dia != dia_anterior):
                   #dia = 0
                   dia_anterior = 0
                   n_rejeitados =0
                   n_aprovados = 0
               
               if resultado_geral =="True":                   
                     n_aprovados +=1
                     self.aprovados.set_status_inicial(str(n_aprovados))

               else:                    
                     n_rejeitados +=1
                     self.rejeitados.set_status_inicial(str(n_rejeitados))
               
               total_pecas = n_aprovados + n_rejeitados

                # Cálculo do yield
               if(total_pecas>0):
                    yield_processo = int((n_aprovados / total_pecas) * 100)
                    self.yield_p.set_status_inicial(str(yield_processo))
                
                    dia_anterior = dia
                     
                    
            
               config['STATUS']['executado'] = "false"
                # Salvar as alterações no arquivo INI
               with open('status.ini', 'w') as configfile:
                 config.write(configfile)

            

            
                
        except requests.exceptions.RequestException:
            # Tratamento para erros de conexão
            self.status_incial.set_status_inicial("Erro de conexão com o servidor")
            
        except Exception as e:
            # Tratamento para outros erros
            self.status_incial.set_status_inicial(f"Erro: {str(e)}")
            
MonitorApp().run()