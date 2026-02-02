"""
Calculadora Científica - Interfaz Gráfica
Desarrollada con tkinter para una experiencia visual profesional
"""

import tkinter as tk
from tkinter import ttk, messagebox
from calculadora import Calculator
import math


class CalculadoraGUI:
    """Interfaz gráfica de la calculadora científica"""
    
    def __init__(self, root):
        """Inicializa la interfaz gráfica"""
        self.root = root
        self.root.title("Calculadora Científica Avanzada")
        self.root.geometry("450x650")
        self.root.resizable(False, False)
        
        # Instancia de la calculadora
        self.calc = Calculator()
        
        # Variables de control
        self.expresion_actual = ""
        self.resultado_mostrado = False
        
        # Configurar estilo
        self.configurar_estilos()
        
        # Crear componentes
        self.crear_display()
        self.crear_frame_principal()
        
    def configurar_estilos(self):
        """Configura los estilos de la aplicación"""
        self.root.configure(bg="#1e1e1e")
        
    def crear_display(self):
        """Crea el área de visualización de resultados"""
        # Frame para el display
        display_frame = tk.Frame(self.root, bg="#1e1e1e", pady=20, padx=10)
        display_frame.pack(fill=tk.BOTH)
        
        # Display de expresión actual
        self.expresion_label = tk.Label(
            display_frame,
            text="",
            font=("Segoe UI", 12),
            bg="#1e1e1e",
            fg="#888888",
            anchor="e",
            height=1
        )
        self.expresion_label.pack(fill=tk.BOTH, padx=10)
        
        # Display de resultado
        self.display = tk.Entry(
            display_frame,
            font=("Segoe UI", 28, "bold"),
            bg="#2d2d2d",
            fg="#ffffff",
            bd=0,
            justify="right",
            insertbackground="#ffffff"
        )
        self.display.pack(fill=tk.BOTH, padx=10, pady=(5, 0), ipady=15)
        self.display.insert(0, "0")
        
    def crear_frame_principal(self):
        """Crea el frame principal que contendrá los botones"""
        self.frame_botones = tk.Frame(self.root, bg="#1e1e1e")
        self.frame_botones.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


def main():
    """Función principal para ejecutar la aplicación"""
    root = tk.Tk()
    app = CalculadoraGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
