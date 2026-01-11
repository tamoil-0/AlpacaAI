def factorial(n):
    """Devuelve el factorial de un número entero no negativo."""
    if n < 0 or not float(n).is_integer():
        return "Error: El factorial solo está definido para enteros no negativos."
    return math.factorial(int(n))
def valor_absoluto(x):
    """Devuelve el valor absoluto de un número."""
    return abs(x)
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
