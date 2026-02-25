from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
import requests
from kivy.config import Config

# Ativar a tela cheia
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
        self.text = "OK"
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
        self.text = "Ferramenta 1"
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
        self.text = "OK"
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
        self.text = "Ferramenta 2"
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
        self.text = "OK"
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
        self.text = "Ferramenta 3"
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status
    
    

class LabelInicializacao(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)  # Desabilitar redimensionamento automático
        self.size = (300, 50)  # Definir tamanho fixo para a Label
        self.pos_hint = {'center_x': 0.5, 'center_y': 0.5}  # Centraliza horizontal e verticalmente

        self.font_size = 100  # Tamanho da fonte
        self.bold = True
        #self.halign = "center"  # Centraliza horizontalmente o texto
        #self.valign = "middle"  # Centraliza verticalmente o texto
        self.color = (1, 1, 1, 1)  # Texto branco (R, G, B, A)
        self.markup = True  # Permite formatação de texto com markup
        self.text = "Conectando ao servidor..."
        #self.bind(size=self.update_rect, pos=self.update_rect)  # Atualiza a borda com a posição/tamanho
    
    def set_status_inicial(self,status):
        
        self.text = status

        
        

class MonitorApp(App):
    def build(self):
        
        layout = FloatLayout()  # Permite posicionamento absolutose
        self.label_inicial = LabelInicializacao()
        self.status_geral = StatusGeral()
        self.status_p1 = StatusParcial1()
        self.label_p1 = LabelParcial1()

        self.status_p2 = StatusParcial2()
        self.label_p2 = LabelParcial2()

        self.status_p3 = StatusParcial3()
        self.label_p3 = LabelParcial3()

        layout.add_widget(self.label_inicial)
        layout.add_widget(self.status_geral)

        layout.add_widget(self.status_p1)
        layout.add_widget(self.label_p1)

        layout.add_widget(self.status_p2)
        layout.add_widget(self.label_p2)

        layout.add_widget(self.status_p3)
        layout.add_widget(self.label_p3)

        # Atualizar o status a cada 1 segundos
        Clock.schedule_interval(self.update_status, 1)

        return layout

    def update_status(self, dt):
        try:
            resultado_p1  = "NOK"
            resultado_p2  = "NOK"
            resultado_p3  = "NOK"
            resultado_GERAL  = "NOK"
            # Fazer a requisição ao servidor Flask no endpoint /executar
            response = requests.get("http://127.0.0.1:6001/executar")
            
            if response.status_code == 200:
                self.label_inicial.set_status_inicial("")
                # Extrair os dados do JSON retornado
                data = response.json()
                
                if "rois_data" in data:
                        rois = data["rois_data"]
                        # Atualizar a interface com base no status das ROIs
                        for roi in rois:
                            roi_index = roi["roi_index"]
                            status = roi["status"]
                            
                            match roi_index:
                                case 1:
                                            # Configurar o texto e a cor com base no status
                                    if status:
                                        resultado_p1 = "OK"                               
                                    else:
                                        resultado_p1 = "NOT_OK"   
                                    self.status_p1.set_status_color(resultado_p1) 

                                case 2:

                                    if status:
                                        resultado_p2 = "OK"                               
                                    else:
                                        resultado_p2 = "NOT_OK"   
                                    self.status_p2.set_status_color(resultado_p2)
                                
                                case 3:

                                    if status:
                                        resultado_p3 = "OK"                               
                                    else:
                                        resultado_p3 = "NOT_OK"   
                                    self.status_p3.set_status_color(resultado_p3) 
                                case _:
                                    print("Caso não encontrado")
                
                        if(resultado_p1 == "OK" and resultado_p2 == "OK" and resultado_p3 =="OK" ):
                            self.status_geral.set_status_color("OK")
                        else:
                            self.status_geral.set_status_color("NOT_OK")
                       
                else:
                    # Se o JSON não contiver 'rois_data'
                    self.status_label.text = "Erro: Dados de ROI ausentes"
                    self.status_label.color = (1, 0.5, 0, 1)  # Laranja
            else:
                # Caso o status da resposta não seja 200
                self.status_label.text = f"Erro no servidor: {response.status_code}"
                self.status_label.color = (1, 0, 0, 1)  # Vermelho
        except requests.exceptions.RequestException:
            # Tratamento para erros de conexão
            self.status_label.text = "Erro de conexão com o servidor"
            self.status_label.color = (1, 0, 0, 1)  # Vermelho
        except Exception as e:
            # Tratamento para outros erros
            self.status_label.text = f"Erro: {str(e)}"
            self.status_label.color = (1, 0, 0, 1)  # Vermelho

MonitorApp().run()

