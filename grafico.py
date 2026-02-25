from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout

class MyApp(App):
    def build(self):
        layout = BoxLayout(orientation="vertical")
        self.label = Label(text="Bem-vindo!")
        layout.add_widget(self.label)

        button = Button(text="Clique aqui")
        button.bind(on_press=self.on_button_click)
        layout.add_widget(button)

        return layout

    def on_button_click(self, instance):
        self.label.text = "Olá, Mundo!"

MyApp().run()

