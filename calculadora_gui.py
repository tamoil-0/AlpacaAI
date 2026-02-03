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
        self.root.geometry("750x650")
        self.root.resizable(False, False)
        
        # Instancia de la calculadora
        self.calc = Calculator()
        
        # Variables de control
        self.expresion_actual = ""
        self.resultado_mostrado = False
        self.modo_grados = True  # True = grados, False = radianes
        self.tema_oscuro = True
        
        # Base de datos de colores
        self.colores = {
            "bg": "#1e1e1e",
            "bg_display": "#1e1e1e",
            "fg_display": "#ffffff",
            "btn_num": "#3a3a3a",
            "btn_op": "#ff8c00",
            "btn_sci": "#2c5f7c",
            "btn_eq": "#4cd964",
            "btn_del": "#ff6b6b",
            "text": "#ffffff"
        }
        
        # Configurar estilo
        self.configurar_estilos()
        
        # Crear componentes
        self.crear_menu_superior()
        self.crear_display()
        self.crear_frame_principal()
        
        # Configurar atajos
        self.configurar_atajos()
        
    def configurar_estilos(self):
        """Configura los estilos de la aplicación"""
        self.root.configure(bg="#1e1e1e")
        
    def crear_menu_superior(self):
        """Crea la barra de menú superior"""
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        
        # Menú Modo
        modo_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Modo", menu=modo_menu)
        modo_menu.add_command(label="Calculadora Científica", command=lambda: None)
        modo_menu.add_command(label="Estadística", command=self.abrir_estadistica)
        modo_menu.add_separator()
        modo_menu.add_command(label="Salir", command=self.root.quit)
        
        # Menú Ver
        ver_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Ver", menu=ver_menu)
        ver_menu.add_command(label="Historial", command=self.mostrar_historial)
        ver_menu.add_separator()
        ver_menu.add_command(label="Cambiar Tema (Claro/Oscuro)", command=self.toggle_tema)
        
        # Menú Ayuda
        ayuda_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Ayuda", menu=ayuda_menu)
        ayuda_menu.add_command(label="Acerca de", command=self.mostrar_acerca_de)

    def mostrar_acerca_de(self):
        """Muestra información sobre la aplicación"""
        messagebox.showinfo(
            "Acerca de",
            "Calculadora Científica Profesional\n"
            "Versión 2.0\n\n"
            "Desarrollada con Python y Tkinter\n"
            "© 2026 Tesis Project"
        )
        
    def configurar_atajos(self):
        """Configura los atajos de teclado"""
        self.root.bind("<Key>", self.manejar_teclado)
        self.root.bind("<Return>", lambda e: self.calcular())
        self.root.bind("<BackSpace>", lambda e: self.borrar_ultimo())
        self.root.bind("<Escape>", lambda e: self.limpiar())
        
    def manejar_teclado(self, event):
        """Maneja la entrada por teclado"""
        char = event.char
        if char.isdigit() or char in ".":
            self.agregar_numero(char)
        elif char in "+-*/":
            mapa = {"*": "×", "/": "÷"}
            self.agregar_operador(mapa.get(char, char))
        elif char in "()":
            self.agregar_caracter(char)
            
    def toggle_tema(self):
        """Cambia entre modo claro y oscuro"""
        self.tema_oscuro = not self.tema_oscuro
        
        if self.tema_oscuro:
            colores = {
                "bg": "#1e1e1e", "btn_num": "#3a3a3a", "text": "#ffffff",
                "bg_display": "#1e1e1e", "fg_display": "#ffffff"
            }
        else:
            colores = {
                "bg": "#f0f0f0", "btn_num": "#e0e0e0", "text": "#000000",
                "bg_display": "#ffffff", "fg_display": "#000000"
            }
            
        self.root.configure(bg=colores["bg"])
        self.frame_botones.configure(bg=colores["bg"])
        self.display.configure(bg=colores["bg_display"], fg=colores["fg_display"])
        
        # Actualizar botones numéricos (simplificado)
        for widget in self.frame_botones.winfo_children():
            if isinstance(widget, tk.Button):
                if widget['text'] in "0123456789.":
                    widget.configure(bg=colores["btn_num"], fg=colores["text"])
    
    def mostrar_historial(self):
        """Muestra ventana con el historial"""
        hist_window = tk.Toplevel(self.root)
        hist_window.title("Historial de Operaciones")
        hist_window.geometry("400x500")
        hist_window.configure(bg="#1e1e1e" if self.tema_oscuro else "#f0f0f0")
        
        # Lista con scrollbar
        scrollbar = tk.Scrollbar(hist_window)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        lista = tk.Listbox(
            hist_window, 
            yscrollcommand=scrollbar.set,
            bg="#2d2d2d" if self.tema_oscuro else "#ffffff",
            fg="#ffffff" if self.tema_oscuro else "#000000",
            font=("Consolas", 12),
            bd=0, highlightthickness=0
        )
        lista.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.config(command=lista.yview)
        
        # Llenar historial
        for item in self.calc.historial:
            lista.insert(tk.END, item)
            
        # Botón limpiar historial
        btn_limpiar = tk.Button(
            hist_window,
            text="Limpiar Historial",
            command=lambda: [self.calc.limpiar_historial(), lista.delete(0, tk.END)],
            bg="#ff6b6b", fg="#ffffff", font=("Segoe UI", 10, "bold"), bd=0, pady=10
        )
        btn_limpiar.pack(fill=tk.X, padx=10, pady=10)
        
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
        # Container principal
        container = tk.Frame(self.root, bg="#1e1e1e")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame científico (izquierda)
        frame_cientifico = tk.Frame(container, bg="#1e1e1e")
        frame_cientifico.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.crear_panel_cientifico(frame_cientifico)
        
        # Frame de botones numéricos (derecha)
        self.frame_botones = tk.Frame(container, bg="#1e1e1e")
        self.frame_botones.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Crear botones
        self.crear_botones_numericos()
    
    def crear_panel_cientifico(self, parent):
        """Crea el panel con funciones científicas"""
        # Configurar grid
        for i in range(3):
            parent.columnconfigure(i, weight=1)
        for i in range(8):
            parent.rowconfigure(i, weight=1)
        
        # Función auxiliar para crear botones científicos con hover
        def crear_btn_cientifico(texto, fila, col, comando, color="#2c5f7c"):
            btn = tk.Button(
                parent,
                text=texto,
                font=("Segoe UI", 11, "bold"),
                bg=color,
                fg="#ffffff",
                bd=0,
                padx=5,
                pady=15,
                cursor="hand2",
                activebackground="#3a7a9c",
                relief=tk.FLAT,
                command=comando
            )
            btn.grid(row=fila, column=col, sticky="nsew", padx=2, pady=2)
            
            # Efectos hover
            def on_enter(e):
                btn['background'] = '#3a7a9c'
            def on_leave(e):
                btn['background'] = color
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
            
            return btn
        
        # Botones trigonométricos
        crear_btn_cientifico("sin", 0, 0, lambda: self.calcular_seno())
        crear_btn_cientifico("cos", 0, 1, lambda: self.calcular_coseno())
        crear_btn_cientifico("tan", 0, 2, lambda: self.calcular_tangente())
        
        # Funciones inversas
        crear_btn_cientifico("sin⁻¹", 1, 0, lambda: self.calcular_arcoseno())
        crear_btn_cientifico("cos⁻¹", 1, 1, lambda: self.calcular_arcocoseno())
        crear_btn_cientifico("tan⁻¹", 1, 2, lambda: self.calcular_arcotangente())
        
        # Logaritmos y exponenciales
        crear_btn_cientifico("log", 2, 0, lambda: self.calcular_log())
        crear_btn_cientifico("ln", 2, 1, lambda: self.calcular_ln())
        crear_btn_cientifico("eˣ", 2, 2, lambda: self.calcular_exp())
        
        # Potencias
        crear_btn_cientifico("xʸ", 3, 0, lambda: self.agregar_operador("^"))
        crear_btn_cientifico("x³", 3, 1, lambda: self.calcular_cubo())
        crear_btn_cientifico("10ˣ", 3, 2, lambda: self.calcular_10_elevado())
        
        # Funciones adicionales
        crear_btn_cientifico("√x", 4, 0, lambda: self.calcular_raiz())
        crear_btn_cientifico("∛x", 4, 1, lambda: self.calcular_raiz_cubica())
        crear_btn_cientifico("|x|", 4, 2, lambda: self.calcular_valor_absoluto())
        
        # Constantes
        crear_btn_cientifico("π", 5, 0, lambda: self.insertar_constante(str(math.pi)), color="#4a4a4a")
        crear_btn_cientifico("e", 5, 1, lambda: self.insertar_constante(str(math.e)), color="#4a4a4a")
        crear_btn_cientifico("%", 5, 2, lambda: self.calcular_porcentaje())
        
        # Modo grados/radianes
        self.btn_modo = tk.Button(
            parent,
            text="DEG",
            font=("Segoe UI", 10, "bold"),
            bg="#505050",
            fg="#ffffff",
            bd=0,
            padx=5,
            pady=15,
            cursor="hand2",
            command=self.cambiar_modo_angulo
        )
        self.btn_modo.grid(row=6, column=0, columnspan=3, sticky="nsew", padx=2, pady=2)
        
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
            relief=tk.FLAT,
            command=comando
        )
        boton.grid(row=fila, column=columna, columnspan=colspan, sticky="nsew", padx=2, pady=2)
        
        # Efectos hover mejorados
        def on_enter(e):
            if color == "#ff6b6b":  # Botones rojos
                boton['background'] = '#ff8585'
            elif color == "#ff8c00":  # Botones naranja
                boton['background'] = '#ffa033'
            elif color == "#4cd964":  # Botón verde
                boton['background'] = '#66e37a'
            elif color == "#505050":  # Botones grises
                boton['background'] = '#656565'
            else:  # Botones normales
                boton['background'] = '#4a4a4a'
        
        def on_leave(e):
            boton['background'] = color
        
        boton.bind("<Enter>", on_enter)
        boton.bind("<Leave>", on_leave)
        
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
    
    # Funciones científicas
    def calcular_seno(self):
        """Calcula el seno del número actual"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.seno(valor, self.modo_grados)
            self.expresion_label.config(text=f"sin({valor})")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_coseno(self):
        """Calcula el coseno del número actual"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.coseno(valor, self.modo_grados)
            self.expresion_label.config(text=f"cos({valor})")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_tangente(self):
        """Calcula la tangente del número actual"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.tangente(valor, self.modo_grados)
            self.expresion_label.config(text=f"tan({valor})")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_arcoseno(self):
        """Calcula el arcoseno del número actual"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.arcoseno(valor, self.modo_grados)
            self.expresion_label.config(text=f"arcsin({valor})")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_arcocoseno(self):
        """Calcula el arcocoseno del número actual"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.arcocoseno(valor, self.modo_grados)
            self.expresion_label.config(text=f"arccos({valor})")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_arcotangente(self):
        """Calcula la arcotangente del número actual"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.arcotangente(valor, self.modo_grados)
            self.expresion_label.config(text=f"arctan({valor})")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_log(self):
        """Calcula el logaritmo base 10"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.logaritmo(valor, 10)
            self.expresion_label.config(text=f"log({valor})")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_ln(self):
        """Calcula el logaritmo natural"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.logaritmo_natural(valor)
            self.expresion_label.config(text=f"ln({valor})")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_exp(self):
        """Calcula e^x"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.exponencial(valor)
            self.expresion_label.config(text=f"e^{valor}")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_cubo(self):
        """Calcula el cubo del número actual"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.potencia(valor, 3)
            self.expresion_label.config(text=f"{valor}³")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_10_elevado(self):
        """Calcula 10^x"""
        try:
            valor = float(self.display.get())
            resultado = 10 ** valor
            self.expresion_label.config(text=f"10^{valor}")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_raiz_cubica(self):
        """Calcula la raíz cúbica"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.raiz_n(valor, 3)
            self.expresion_label.config(text=f"∛{valor}")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_valor_absoluto(self):
        """Calcula el valor absoluto"""
        try:
            valor = float(self.display.get())
            resultado = self.calc.valor_absoluto(valor)
            self.expresion_label.config(text=f"|{valor}|")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def calcular_porcentaje(self):
        """Calcula el porcentaje"""
        try:
            valor = float(self.display.get())
            resultado = valor / 100
            self.expresion_label.config(text=f"{valor}%")
            self.display.delete(0, tk.END)
            self.display.insert(0, str(resultado))
            self.resultado_mostrado = True
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def insertar_constante(self, valor):
        """Inserta una constante matemática"""
        if self.resultado_mostrado:
            self.display.delete(0, tk.END)
            self.resultado_mostrado = False
        self.display.insert(tk.END, valor)
    
    def cambiar_modo_angulo(self):
        """Cambia entre grados y radianes"""
        self.modo_grados = not self.modo_grados
        self.btn_modo.config(text="DEG" if self.modo_grados else "RAD")

    def abrir_estadistica(self):
        """Abre el módulo de estadísticas"""
        ventana = tk.Toplevel(self.root)
        ventana.title("Estadística")
        ventana.geometry("400x500")
        ventana.configure(bg=self.colores["bg"])
        
        # Etiqueta de instrucciones
        tk.Label(
            ventana,
            text="Ingrese números separados por comas:",
            font=("Segoe UI", 10),
            bg=self.colores["bg"],
            fg=self.colores["text"]
        ).pack(pady=10)
        
        # Área de entrada
        entrada = tk.Entry(
            ventana,
            font=("Segoe UI", 12),
            bg="#2d2d2d",
            fg="#ffffff"
        )
        entrada.pack(fill=tk.X, padx=20)
        entrada.focus()
        
        # Área de resultados
        resultados_frame = tk.Frame(ventana, bg=self.colores["bg"])
        resultados_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        lbl_res = tk.Label(
            resultados_frame,
            text="",
            font=("Consolas", 11),
            justify=tk.LEFT,
            bg=self.colores["bg"],
            fg=self.colores["text"]
        )
        lbl_res.pack()
        
        def calcular():
            try:
                texto = entrada.get()
                # Permitir comas o espacios
                texto = texto.replace(" ", ",")
                numeros = [float(x) for x in texto.split(",") if x.strip()]
                
                if not numeros:
                    return
                
                media = self.calc.media(numeros)
                mediana = self.calc.mediana(numeros)
                stdev = self.calc.desviacion_estandar(numeros)
                
                res_texto = (
                    f"Cantidad:    {len(numeros)}\n"
                    f"Suma:        {sum(numeros):.4f}\n"
                    f"Mínimo:      {min(numeros):.4f}\n"
                    f"Máximo:      {max(numeros):.4f}\n"
                    f"--------------------\n"
                    f"Media:       {media:.4f}\n"
                    f"Mediana:     {mediana:.4f}\n"
                    f"Desv. Est.:  {stdev:.4f}"
                )
                lbl_res.config(text=res_texto)
                
            except Exception as e:
                messagebox.showerror("Error", f"Datos inválidos: {str(e)}")
        
        # Botón Calcular
        tk.Button(
            ventana,
            text="Calcular Estadísticas",
            command=calcular,
            bg=self.colores["btn_op"],
            fg="#ffffff",
            font=("Segoe UI", 11, "bold"),
            bd=0, pady=10, cursor="hand2"
        ).pack(fill=tk.X, padx=20, pady=(0, 20))


def main():
    """Función principal para ejecutar la aplicación"""
    root = tk.Tk()
    app = CalculadoraGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
