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

status_largura = 60
Status_altura  = 40
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
        self.size = (status_largura, Status_altura)  # Definir largura e altura fixas
        self.pos = (10, 800)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.1, 0.1, 1)  # Cor inicial do fundo (verde escuro)
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
        self.size = (40, 30)  # Definir largura e altura fixas
        self.pos = (100, 800)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 25 # Tamanho da fonte
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
        self.size = (status_largura, Status_altura) # Definir largura e altura fixas
        self.pos = (10, 750)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.1, 0.1, 1)   # Cor inicial do fundo (verde escuro)
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
        self.size = (40, 30)  # Definir largura e altura fixas
        self.pos = (100, 750)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 25  # Tamanho da fonte
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
        self.size = (status_largura, Status_altura)  # Definir largura e altura fixas
        self.pos = (10, 700)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.1, 0.1, 1)   # Cor inicial do fundo (verde escuro)
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
        self.size = (40, 30)  # Definir largura e altura fixas
        self.pos = (100, 700)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 25  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "ROI 3"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status


class StatusParcial4(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (status_largura, Status_altura)  # Definir largura e altura fixas
        self.pos = (10, 650)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.1, 0.1, 1)  # Cor inicial do fundo (verde escuro)
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

class LabelParcial4(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (40, 30)  # Definir largura e altura fixas
        self.pos = (100, 650)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 25  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "ROI 4"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status


class StatusParcial5(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (status_largura, Status_altura)  # Definir largura e altura fixas
        self.pos = (10, 600)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.1, 0.1, 1)  # Cor inicial do fundo (verde escuro)
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

class LabelParcial5(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (40, 30)  # Definir largura e altura fixas
        self.pos = (100, 600)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 25  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "ROI 5"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status


class StatusParcial6(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (status_largura, Status_altura)  # Definir largura e altura fixas
        self.pos = (10, 550)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.1, 0.1, 1)  # Cor inicial do fundo (verde escuro)
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

class LabelParcial6(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (40, 30)  # Definir largura e altura fixas
        self.pos = (100,550)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 25  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "ROI 6"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class StatusParcial7(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (status_largura, Status_altura)  # Definir largura e altura fixas
        self.pos = (10, 500)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.1, 0.1, 1)  # Cor inicial do fundo (verde escuro)
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

class LabelParcial7(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (40, 30)  # Definir largura e altura fixas
        self.pos = (100,500)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 25  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "ROI 7"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class StatusParcial8(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (status_largura, Status_altura)  # Definir largura e altura fixas
        self.pos = (10, 450)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.1, 0.1, 1)  # Cor inicial do fundo (verde escuro)
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

class LabelParcial8(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (40, 30)  # Definir largura e altura fixas
        self.pos = (100,450)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 25  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "ROI 8"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class StatusParcial9(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (status_largura, Status_altura)  # Definir largura e altura fixas
        self.pos = (10, 400)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.1, 0.1, 1)  # Cor inicial do fundo (verde escuro)
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

class LabelParcial9(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (40, 30)  # Definir largura e altura fixas
        self.pos = (100,400)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 25  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "ROI 9"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class StatusParcial10(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (status_largura, Status_altura)  # Definir largura e altura fixas
        self.pos = (10, 350)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.1, 0.1, 1)  # Cor inicial do fundo (verde escuro)
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

class LabelParcial10(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (40, 30)  # Definir largura e altura fixas
        self.pos = (100,350)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 25  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "ROI 10"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class StatusParcial11(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (status_largura, Status_altura)  # Definir largura e altura fixas
        self.pos = (10, 300)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.1, 0.1, 1)  # Cor inicial do fundo (verde escuro)
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

class LabelParcial11(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (40, 30)  # Definir largura e altura fixas
        self.pos = (100,300)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 25  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "ROI 11"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class StatusParcial12(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (status_largura, Status_altura)  # Definir largura e altura fixas
        self.pos = (10, 250)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30  # Tamanho da fonte
        self.bold = True
        self.halign = "center"  # Centraliza horizontalmente o texto
        self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "**"
        self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho

        # Desenhar o fundo e a borda
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.1, 0.1, 1)  # Cor inicial do fundo (verde escuro)
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

class LabelParcial12(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (40, 30)  # Definir largura e altura fixas
        self.pos = (100,250)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 25  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "ROI 12"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status


class LabelTotalPecas(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (10, 180)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 26 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (0, 0, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "Total:"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class TotalPecas(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (80, 180)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 26 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (0, 0, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = ""
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status




class LabelAprovados(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (10, 150)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 26 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (0, 1, 0, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "Pass:"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class Aprovados(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (80, 150)  # Canto superior esquerdo (ajuste conforme resolução)
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
        self.pos = (5, 120)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 26 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 0, 0, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "Fail:"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class Rejeitados(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (80, 120)  # Canto superior esquerdo (ajuste conforme resolução)
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
        self.pos = (10, 90)  # Canto superior esquerdo (ajuste conforme resolução)
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
        self.pos = (80, 90)  # Canto superior esquerdo (ajuste conforme resolução)
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


class LabelTemperatura(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (20, 20)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 18 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "Temp. CPU:"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class Temperatura(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (110, 20)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 18 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = ""
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status
    
class LabelTempoTotal(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (1600, 90)  # Canto superior esquerdo (ajuste conforme resolução)
        

        self.font_size = 26 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "T.T: "
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

class TempoTotal(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (1690, 90)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 26 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = ""
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status  


class LabelPrograma(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (1580, 1000)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (0, 1, 1, 1)  # Ciano (R=0, G=1, B=1, A=1)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "Programa: "
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status
class programa(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar ajuste automático de tamanho
        self.size = (60, 50)  # Definir largura e altura fixas
        self.pos = (1700, 1000)  # Canto superior esquerdo (ajuste conforme resolução)
        self.font_size = 30 # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (0, 1, 1, 1)  # Ciano (R=0, G=1, B=1, A=1)
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
        

def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as file:
            temp = int(file.read()) / 1000.0  # O valor vem em milicelsius
            return temp
    except FileNotFoundError:
        return None  # Retorna None se o arquivo não for encontrado

n_aprovados = 0
n_rejeitados = 0

n_programa_anterior = "0"

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

        self.status_p4 = StatusParcial4()
        self.label_p4 = LabelParcial4()
            
        self.status_p5 = StatusParcial5()
        self.label_p5 = LabelParcial5()
        
        self.status_p6 = StatusParcial6()
        self.label_p6 = LabelParcial6()

        self.status_p7 = StatusParcial7()
        self.label_p7 = LabelParcial7()
        
        self.status_p8 = StatusParcial8()
        self.label_p8 = LabelParcial8()
          
        self.status_p9 = StatusParcial9()
        self.label_p9 = LabelParcial9()

        self.status_p10 = StatusParcial10()
        self.label_p10 = LabelParcial10()
        
        self.status_p11 = StatusParcial11()
        self.label_p11 = LabelParcial11()
          
        self.status_p12 = StatusParcial12()
        self.label_p12 = LabelParcial12()
             
        self.label_total_pecas = LabelTotalPecas()
        self.total_pecas = TotalPecas()

        self.label_aprovados = LabelAprovados()
        self.aprovados = Aprovados()

        self.label_TempoTotal = LabelTempoTotal()
        self.TempoTotal = TempoTotal()
        
        self.label_rejeitados = LabelRejeitados()
        self.rejeitados = Rejeitados()

        self.label_yield = LabelYield()
        self.yield_p = Yield()

        self.label_Temperatura = LabelTemperatura()
        self.Temperatura = Temperatura()

        self.label_programa = LabelPrograma()
        self.programa = programa()

        layout.add_widget(self.status_incial)
        layout.add_widget(self.status_geral)

        layout.add_widget(self.status_p1)
        layout.add_widget(self.label_p1)

        layout.add_widget(self.status_p2)
        layout.add_widget(self.label_p2)

        layout.add_widget(self.status_p3)
        layout.add_widget(self.label_p3)


        layout.add_widget(self.status_p4)
        layout.add_widget(self.label_p4)

        layout.add_widget(self.status_p5)
        layout.add_widget(self.label_p5)

        layout.add_widget(self.status_p6)
        layout.add_widget(self.label_p6)

        layout.add_widget(self.status_p7)
        layout.add_widget(self.label_p7)

       

        layout.add_widget(self.status_p8)
        layout.add_widget(self.label_p8)

        layout.add_widget(self.status_p9)
        layout.add_widget(self.label_p9)

        layout.add_widget(self.status_p10)
        layout.add_widget(self.label_p10)

        layout.add_widget(self.status_p11)
        layout.add_widget(self.label_p11)

        layout.add_widget(self.status_p12)
        layout.add_widget(self.label_p12)




        layout.add_widget(self.imagem)
        
        layout.add_widget(self.label_total_pecas)
        layout.add_widget(self.total_pecas)


        layout.add_widget(self.label_aprovados)
        layout.add_widget(self.aprovados)

        layout.add_widget(self.label_rejeitados)
        layout.add_widget(self.rejeitados)

        layout.add_widget(self.label_yield)
        layout.add_widget(self.yield_p)

        layout.add_widget(self.label_Temperatura)
        layout.add_widget(self.Temperatura)

        layout.add_widget(self.label_TempoTotal)
        layout.add_widget(self.TempoTotal)

        layout.add_widget(self.label_programa)
        layout.add_widget(self.programa)

        for i in range(1, 13):
       
            getattr(self, f'status_p{i}').opacity = 0
            getattr(self, f'status_p{i}').disabled = True
            getattr(self, f'label_p{i}').opacity = 0
            getattr(self, f'label_p{i}').disabled = True
            
            


        # Atualizar o status a cada 0.2 segundos
        
        Clock.schedule_interval(self.update_status, 0.2)

        return layout
    
  

    def update_status(self, dt):
        try:
            agora = datetime.now()
            global n_aprovados
            global n_rejeitados
            global dia
            global dia_anterior
            global n_programa_anterior
            dia = agora.day
           
           
            # Ler o arquivo de configuração INI
            config = configparser.ConfigParser()
            config.read('status.ini')

            # Obter os parâmetros de ajuste de imagem
            executado = str_to_bool(config['STATUS']['executado'])
            #executado    = str(config['STATUS']['executado'])
            resultado_p1 = str(config['STATUS']['parcial_p1'])
            resultado_p2 = str(config['STATUS']['parcial_p2'])
            resultado_p3 = str(config['STATUS']['parcial_p3'])
            resultado_p4 = str(config['STATUS']['parcial_p4'])
            resultado_p5 = str(config['STATUS']['parcial_p5'])
            resultado_p6 = str(config['STATUS']['parcial_p6'])
            resultado_p7 = str(config['STATUS']['parcial_p7'])
            resultado_p8 = str(config['STATUS']['parcial_p8'])
            resultado_p9 = str(config['STATUS']['parcial_p9'])
            resultado_p10 = str(config['STATUS']['parcial_p10'])
            resultado_p11 = str(config['STATUS']['parcial_p11'])
            resultado_p12 = str(config['STATUS']['parcial_p12'])
            resultado_geral = str(config['STATUS']['status_geral'])
            TempoTotal = str(config['STATUS']['tempo_execucao'])
            n_rois = int(config['STATUS']['n_rois'])
            n_programa = str(config['STATUS']['programa'])
            

            if(executado== True):
               # Aqui, tente imprimir para verificar se o caminho está correto
               print(f"Atualizando imagem com o caminho: imagem_status.jpg")
               Clock.schedule_once(lambda dt: self.imagem.set_image('imagem_status.jpg'))
               for i in range(1, n_rois+1):
                  
                  getattr(self, f'status_p{i}').opacity = 1
                  getattr(self, f'status_p{i}').disabled = False
                  getattr(self, f'label_p{i}').opacity = 1
                  getattr(self, f'label_p{i}').disabled = False

               
                   
                  if(i ==1): 
                       
                        if(resultado_p1 =="True"):
                            self.status_p1.set_status_color("OK") 
                            
                        else:
                            self.status_p1.set_status_color("NOT_OK") 
                  if(i ==2): 
                                
                        if(resultado_p2 =="True"):
                            self.status_p2.set_status_color("OK") 
                        else:
                            self.status_p2.set_status_color("NOT_OK") 
                  if(i ==3):          
                        if(resultado_p3 =="True"):
                            self.status_p3.set_status_color("OK") 
                        else:
                            self.status_p3.set_status_color("NOT_OK") 
                  if(i ==4):           
                        if(resultado_p4 =="True"):
                            self.status_p4.set_status_color("OK") 
                        else:
                            self.status_p4.set_status_color("NOT_OK") 
                  if(i ==5):           
                        if(resultado_p5 =="True"):
                            self.status_p5.set_status_color("OK") 
                        else:
                            self.status_p5.set_status_color("NOT_OK") 
                  if(i ==6):           
                        if(resultado_p6 =="True"):
                            self.status_p6.set_status_color("OK") 
                        else:
                            self.status_p6.set_status_color("NOT_OK") 
                  if(i ==7):           
                        if(resultado_p7 =="True"):
                            self.status_p7.set_status_color("OK") 
                        else:
                            self.status_p7.set_status_color("NOT_OK") 
                  if(i ==8):           
                        if(resultado_p8 =="True"):
                            self.status_p8.set_status_color("OK") 
                        else:
                            self.status_p8.set_status_color("NOT_OK") 
                  if(i ==9):           
                        if(resultado_p9 =="True"):
                            self.status_p9.set_status_color("OK") 
                        else:
                            self.status_p9.set_status_color("NOT_OK") 
                  if(i ==10):           
                        if(resultado_p10 =="True"):
                            self.status_p10.set_status_color("OK") 
                        else:
                            self.status_p10.set_status_color("NOT_OK") 
                  if(i ==11):           
                        if(resultado_p11 =="True"):
                            self.status_p11.set_status_color("OK") 
                        else:
                            self.status_p11.set_status_color("NOT_OK") 
                  if(i ==12):           
                        if(resultado_p12 =="True"):
                            self.status_p12.set_status_color("OK") 
                        else:
                            self.status_p12.set_status_color("NOT_OK") 

                  if(resultado_geral =="True"):
                        self.status_geral.set_status_color("OK") 
                  else:
                        self.status_geral.set_status_color("NOT_OK") 

               i = n_rois +1
               for i in range(i,13):
                  getattr(self, f'status_p{i}').opacity = 0
                  getattr(self, f'status_p{i}').disabled = True
                  getattr(self, f'label_p{i}').opacity = 0
                  getattr(self, f'label_p{i}').disabled = True
                   
            
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
               self.total_pecas.set_status_inicial(str(total_pecas))
                # Cálculo do yield
               if(total_pecas>0):
                    yield_processo = int((n_aprovados / total_pecas) * 100)
                    self.yield_p.set_status_inicial(str(yield_processo))
                
                    dia_anterior = dia

               TempoTotal =f"{TempoTotal} ms"
               self.TempoTotal.set_status_inicial(TempoTotal)  
               self.programa.set_status_inicial(n_programa)  
                # Obter a temperatura da CPU
               temp = get_cpu_temperature()
               temp =  f"{temp:.2f}°C"
               self.Temperatura.set_status_inicial(temp)
                    
               
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
def str_to_bool(value):
    return str(value).strip().lower() in ['true', '1', 'yes', 'sim', 'on']
            
MonitorApp().run()