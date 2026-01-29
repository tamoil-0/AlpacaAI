"""
Módulo base de la Calculadora Científica Avanzada
"""

import math
import numpy as np
from typing import Union, Optional

class Calculator:
    """
    Clase base para operaciones de calculadora científica.
    """

    def __init__(self):
        self.history = []
        self.memory = 0

    def add_to_history(self, operation: str, result: Union[float, complex]):
        """Añade una operación al historial."""
        self.history.append({
            'operation': operation,
            'result': result,
            'timestamp': np.datetime64('now')
        })

    def clear_history(self):
        """Limpia el historial."""
        self.history = []

    # Operaciones básicas
    def add(self, a: float, b: float) -> float:
        result = a + b
        self.add_to_history(f"{a} + {b}", result)
        return result

    def subtract(self, a: float, b: float) -> float:
        result = a - b
        self.add_to_history(f"{a} - {b}", result)
        return result

    def multiply(self, a: float, b: float) -> float:
        result = a * b
        self.add_to_history(f"{a} * {b}", result)
        return result

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("No se puede dividir por cero")
        result = a / b
        self.add_to_history(f"{a} / {b}", result)
        return result

    # Funciones científicas básicas
    def sin(self, x: float, degrees: bool = False) -> float:
        if degrees:
            x = math.radians(x)
        result = math.sin(x)
        self.add_to_history(f"sin({x})", result)
        return result

    def cos(self, x: float, degrees: bool = False) -> float:
        if degrees:
            x = math.radians(x)
        result = math.cos(x)
        self.add_to_history(f"cos({x})", result)
        return result

    def tan(self, x: float, degrees: bool = False) -> float:
        if degrees:
            x = math.radians(x)
        result = math.tan(x)
        self.add_to_history(f"tan({x})", result)
        return result

    def log(self, x: float, base: Optional[float] = None) -> float:
        if x <= 0:
            raise ValueError("El logaritmo requiere un número positivo")
        if base is None:
            result = math.log(x)
            self.add_to_history(f"ln({x})", result)
        else:
            result = math.log(x, base)
            self.add_to_history(f"log_{base}({x})", result)
        return result

    def exp(self, x: float) -> float:
        result = math.exp(x)
        self.add_to_history(f"e^{x}", result)
        return result

    def power(self, base: float, exponent: float) -> float:
        result = math.pow(base, exponent)
        self.add_to_history(f"{base}^{exponent}", result)
        return result

    def sqrt(self, x: float) -> float:
        if x < 0:
            raise ValueError("Raíz cuadrada de número negativo")
        result = math.sqrt(x)
        self.add_to_history(f"√{x}", result)
        return result

    # Memoria
    def memory_store(self, value: float):
        self.memory = value

    def memory_recall(self) -> float:
        return self.memory

    def memory_clear(self):
        self.memory = 0

    def memory_add(self, value: float):
        self.memory += value

    def memory_subtract(self, value: float):
        self.memory -= value