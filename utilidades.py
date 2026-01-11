"""
Archivo de utilidades matemáticas
Autor: Jhon (mejoras automáticas)
Fecha: 2026-01-11
"""

import math

def cuadrado(x):
    """Devuelve el cuadrado de un número."""
    return x * x

def cubo(x):
    """Devuelve el cubo de un número."""
    return x * x * x

def raiz_cuadrada(x):
    """Devuelve la raíz cuadrada de un número."""
    if x < 0:
        return "Error: No se puede calcular la raíz cuadrada de un número negativo."
    return math.sqrt(x)
