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
        
        # Crear botones
        self.crear_botones_numericos()
        
    def crear_boton(self, texto, fila, columna, comando, colspan=1, color="#3a3a3a", texto_color="#ffffff"):
        """Crea un botón personalizado"""
        boton = tk.Button(
            self.frame_botones,
            text=texto,
            font=("Segoe UI", 14, "bold"),
            bg=color,
            fg=texto_color,
            bd=0,
            padx=10,
            pady=20,
            cursor="hand2",
            activebackground="#4a4a4a",
            activeforeground="#ffffff",
            command=comando
        )
        boton.grid(row=fila, column=columna, columnspan=colspan, sticky="nsew", padx=2, pady=2)
        
        # Configurar peso de las columnas y filas
        for i in range(5):
            self.frame_botones.columnconfigure(i, weight=1)
        for i in range(6):
            self.frame_botones.rowconfigure(i, weight=1)
        
        return boton
    
    def crear_botones_numericos(self):
        """Crea la grilla completa de botones"""
        # Fila 0: Funciones especiales
        self.crear_boton("C", 0, 0, lambda: self.limpiar(), color="#ff6b6b")
        self.crear_boton("⌫", 0, 1, lambda: self.borrar_ultimo(), color="#ff6b6b")
        self.crear_boton("(", 0, 2, lambda: self.agregar_caracter("("), color="#505050")
        self.crear_boton(")", 0, 3, lambda: self.agregar_caracter(")"), color="#505050")
        self.crear_boton("÷", 0, 4, lambda: self.agregar_operador("÷"), color="#ff8c00")
        
        # Fila 1
        self.crear_boton("7", 1, 0, lambda: self.agregar_numero("7"))
        self.crear_boton("8", 1, 1, lambda: self.agregar_numero("8"))
        self.crear_boton("9", 1, 2, lambda: self.agregar_numero("9"))
        self.crear_boton("×", 1, 3, lambda: self.agregar_operador("×"), color="#ff8c00")
        self.crear_boton("x²", 1, 4, lambda: self.calcular_cuadrado(), color="#505050")
        
        # Fila 2
        self.crear_boton("4", 2, 0, lambda: self.agregar_numero("4"))
        self.crear_boton("5", 2, 1, lambda: self.agregar_numero("5"))
        self.crear_boton("6", 2, 2, lambda: self.agregar_numero("6"))
        self.crear_boton("-", 2, 3, lambda: self.agregar_operador("-"), color="#ff8c00")
        self.crear_boton("√", 2, 4, lambda: self.calcular_raiz(), color="#505050")
        
        # Fila 3
        self.crear_boton("1", 3, 0, lambda: self.agregar_numero("1"))
        self.crear_boton("2", 3, 1, lambda: self.agregar_numero("2"))
        self.crear_boton("3", 3, 2, lambda: self.agregar_numero("3"))
        self.crear_boton("+", 3, 3, lambda: self.agregar_operador("+"), color="#ff8c00")
        self.crear_boton("x!", 3, 4, lambda: self.calcular_factorial(), color="#505050")
        
        # Fila 4
        self.crear_boton("±", 4, 0, lambda: self.cambiar_signo(), color="#505050")
        self.crear_boton("0", 4, 1, lambda: self.agregar_numero("0"))
        self.crear_boton(".", 4, 2, lambda: self.agregar_numero("."))
        self.crear_boton("=", 4, 3, lambda: self.calcular(), colspan=2, color="#4cd964")
    
    # Funciones de entrada
    def agregar_numero(self, numero):
        """Agrega un número al display"""
        if self.resultado_mostrado:
            self.display.delete(0, tk.END)
            self.resultado_mostrado = False
        
        actual = self.display.get()
        if actual == "0":
            self.display.delete(0, tk.END)
        self.display.insert(tk.END, numero)
    
    def agregar_operador(self, operador):
        """Agrega un operador al display"""
        self.resultado_mostrado = False
        self.display.insert(tk.END, f" {operador} ")
    
    def agregar_caracter(self, caracter):
        """Agrega un carácter especial"""
        if self.resultado_mostrado:
            self.display.delete(0, tk.END)
            self.resultado_mostrado = False
        self.display.insert(tk.END, caracter)
    
    def limpiar(self):
        """Limpia el display"""
        self.display.delete(0, tk.END)
        self.display.insert(0, "0")
        self.expresion_label.config(text="")
        self.resultado_mostrado = False
    
    def borrar_ultimo(self):
        """Borra el último carácter"""
        actual = self.display.get()
        if len(actual) > 1:
            self.display.delete(len(actual)-1, tk.END)
        else:
            self.display.delete(0, tk.END)
            self.display.insert(0, "0")
    
    def cambiar_signo(self):
        """Cambia el signo del número actual"""
        try:
            valor = float(self.display.get())
            self.display.delete(0, tk.END)
            self.display.insert(0, str(-valor))
        except:
            pass
    
    def calcular(self):
        """Calcula el resultado de la expresión"""
        try:
            expresion = self.display.get()
            # Reemplazar símbolos visuales por operadores Python
            expresion = expresion.replace("×", "*").replace("÷", "/")
            resultado = eval(expresion)
            
            self.expresion_label.config(text=self.display.get())
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", "Expresión inválida")
    
    def calcular_cuadrado(self):
        """Calcula el cuadrado del número actual"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.potencia(valor, 2)
            self.expresion_label.config(text=f"{valor}²")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_raiz(self):
        """Calcula la raíz cuadrada del número actual"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.raiz_cuadrada(valor)
            self.expresion_label.config(text=f"√{valor}")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_factorial(self):
        """Calcula el factorial del número actual"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.factorial(valor)
            self.expresion_label.config(text=f"{int(valor)}!")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    """Función principal para ejecutar la aplicación"""
    root = tk.Tk()
    app = CalculadoraGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
