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

