def tangente(x):
    """Devuelve la tangente de un ángulo en radianes."""
    return math.tan(x)
def coseno(x):
    """Devuelve el coseno de un ángulo en radianes."""
    return math.cos(x)
def seno(x):
    """Devuelve el seno de un ángulo en radianes."""
    return math.sin(x)
def log_base_10(x):
    """Devuelve el logaritmo en base 10 de un número positivo."""
    if x <= 0:
        return "Error: El logaritmo solo está definido para números positivos."
    return math.log10(x)
def mcm(a, b):
    """Devuelve el mínimo común múltiplo de dos números enteros."""
    a, b = int(a), int(b)
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)
def mcd(a, b):
    """Devuelve el máximo común divisor de dos números enteros."""
    return math.gcd(int(a), int(b))
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
