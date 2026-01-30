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


