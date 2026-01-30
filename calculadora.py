# Calculadora Científica Avanzada
# Proyecto de Desarrollo - Fase 1: Fundación

"""
Calculadora científica con funciones básicas y avanzadas
Desarrollada como parte del plan de desarrollo de 2 meses
"""

import math
import sys


class Calculator:
    """Clase base de la calculadora científica"""
    
    def __init__(self):
        """Inicializa la calculadora"""
        self.resultado = 0
        self.historial = []
    
    def sumar(self, a, b):
        """Suma dos números"""
        resultado = a + b
        self.historial.append(f"{a} + {b} = {resultado}")
        return resultado
    
    def restar(self, a, b):
        """Resta dos números"""
        resultado = a - b
        self.historial.append(f"{a} - {b} = {resultado}")
        return resultado
    
    def multiplicar(self, a, b):
        """Multiplica dos números"""
        resultado = a * b
        self.historial.append(f"{a} × {b} = {resultado}")
        return resultado
    
    def dividir(self, a, b):
        """Divide dos números"""
        if b == 0:
            raise ValueError("No se puede dividir por cero")
        resultado = a / b
        self.historial.append(f"{a} ÷ {b} = {resultado}")
        return resultado
    
    def potencia(self, base, exponente):
        """Calcula la potencia de un número"""
        resultado = base ** exponente
        self.historial.append(f"{base}^{exponente} = {resultado}")
        return resultado
    
    def raiz_cuadrada(self, n):
        """Calcula la raíz cuadrada"""
        if n < 0:
            raise ValueError("No se puede calcular raíz cuadrada de número negativo")
        resultado = math.sqrt(n)
        self.historial.append(f"√{n} = {resultado}")
        return resultado
    
    def factorial(self, n):
        """Calcula el factorial de un número"""
        if n < 0:
            raise ValueError("No se puede calcular factorial de número negativo")
        if n > 170:
            raise ValueError("Número demasiado grande para factorial")
        resultado = math.factorial(int(n))
        self.historial.append(f"{n}! = {resultado}")
        return resultado
    
    def seno(self, angulo, grados=True):
        """Calcula el seno de un ángulo"""
        if grados:
            angulo = math.radians(angulo)
        resultado = math.sin(angulo)
        unidad = "°" if grados else "rad"
        self.historial.append(f"sin({angulo}{unidad}) = {resultado}")
        return resultado
    
    def coseno(self, angulo, grados=True):
        """Calcula el coseno de un ángulo"""
        if grados:
            angulo = math.radians(angulo)
        resultado = math.cos(angulo)
        unidad = "°" if grados else "rad"
        self.historial.append(f"cos({angulo}{unidad}) = {resultado}")
        return resultado
    
    def tangente(self, angulo, grados=True):
        """Calcula la tangente de un ángulo"""
        if grados:
            angulo = math.radians(angulo)
        resultado = math.tan(angulo)
        unidad = "°" if grados else "rad"
        self.historial.append(f"tan({angulo}{unidad}) = {resultado}")
        return resultado


# Función principal de demostración
if __name__ == "__main__":
    print("=== Calculadora Científica Avanzada ===")
    calc = Calculator()
    
    # Demostración de operaciones básicas
    print("\n--- Operaciones Básicas ---")
    print(f"5 + 3 = {calc.sumar(5, 3)}")
    print(f"10 - 4 = {calc.restar(10, 4)}")
    print(f"6 × 7 = {calc.multiplicar(6, 7)}")
    print(f"20 ÷ 4 = {calc.dividir(20, 4)}")
    
    # Demostración de funciones científicas
    print("\n--- Funciones Científicas ---")
    print(f"2^8 = {calc.potencia(2, 8)}")
    print(f"√16 = {calc.raiz_cuadrada(16)}")
    print(f"5! = {calc.factorial(5)}")
    
    # Demostración de trigonometría
    print("\n--- Funciones Trigonométricas ---")
    print(f"sin(30°) = {calc.seno(30)}")
    print(f"cos(60°) = {calc.coseno(60)}")
    print(f"tan(45°) = {calc.tangente(45)}")
    
    # Mostrar historial
    print("\n--- Historial de Operaciones ---")
    for operacion in calc.historial:
        print(f"  {operacion}")





