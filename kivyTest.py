
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
import requests

class MonitorApp(App):
    def build(self):
        self.layout = BoxLayout(orientation="vertical", padding=20, spacing=20)
        
        # Label para exibir o status (ativando a marcação de texto)
        self.status_label = Label(
            text="Conectando ao servidor...", 
            font_size=46, 
            markup=True  # Ativa suporte a cores e formatação
        )
        self.layout.add_widget(self.status_label)

        # Atualizar o status a cada 2 segundos
        Clock.schedule_interval(self.update_status, 1)

        return self.layout

    def update_status(self, dt):
        try:
            # Fazer a requisição ao servidor Flask
            response = requests.get("http://127.0.0.1:5000/status")
            if response.status_code == 200:
                status = response.json().get("status", "Desconhecido")
                if status == "OK":
                    self.status_label.text = "[color=00FF00]Status: OK[/color]"
                elif status == "NOT_OK":
                    self.status_label.text = "[color=FF0000]Status: NÃO OK[/color]"
                else:
                    self.status_label.text = "Status: Desconhecido"
            else:
                self.status_label.text = "Erro no servidor"
        except requests.exceptions.RequestException:
            self.status_label.text = "Falha na conexão com o servidor"

MonitorApp().run()
