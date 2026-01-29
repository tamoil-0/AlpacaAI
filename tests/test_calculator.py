"""
Tests básicos para la clase Calculator
"""

import unittest
import math
from src.calculator import Calculator

class TestCalculator(unittest.TestCase):
    """Clase de pruebas para Calculator."""

    def setUp(self):
        """Configuración antes de cada test."""
        self.calc = Calculator()

    def test_add(self):
        """Prueba de suma."""
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)
        self.assertEqual(len(self.calc.history), 1)

    def test_subtract(self):
        """Prueba de resta."""
        result = self.calc.subtract(5, 3)
        self.assertEqual(result, 2)

    def test_multiply(self):
        """Prueba de multiplicación."""
        result = self.calc.multiply(4, 3)
        self.assertEqual(result, 12)

    def test_divide(self):
        """Prueba de división."""
        result = self.calc.divide(10, 2)
        self.assertEqual(result, 5)

    def test_divide_by_zero(self):
        """Prueba de división por cero."""
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)

    def test_sin_radians(self):
        """Prueba de seno en radianes."""
        result = self.calc.sin(0)
        self.assertAlmostEqual(result, 0)

    def test_sin_degrees(self):
        """Prueba de seno en grados."""
        result = self.calc.sin(90, degrees=True)
        self.assertAlmostEqual(result, 1)

    def test_cos(self):
        """Prueba de coseno."""
        result = self.calc.cos(0)
        self.assertAlmostEqual(result, 1)

    def test_tan(self):
        """Prueba de tangente."""
        result = self.calc.tan(0)
        self.assertAlmostEqual(result, 0)

    def test_log_natural(self):
        """Prueba de logaritmo natural."""
        result = self.calc.log(math.e)
        self.assertAlmostEqual(result, 1)

    def test_log_base_10(self):
        """Prueba de logaritmo base 10."""
        result = self.calc.log(100, 10)
        self.assertAlmostEqual(result, 2)

    def test_exp(self):
        """Prueba de exponencial."""
        result = self.calc.exp(0)
        self.assertAlmostEqual(result, 1)

    def test_power(self):
        """Prueba de potencia."""
        result = self.calc.power(2, 3)
        self.assertEqual(result, 8)

    def test_sqrt(self):
        """Prueba de raíz cuadrada."""
        result = self.calc.sqrt(9)
        self.assertEqual(result, 3)

    def test_memory_operations(self):
        """Prueba de operaciones de memoria."""
        self.calc.memory_store(5)
        self.assertEqual(self.calc.memory_recall(), 5)

        self.calc.memory_add(3)
        self.assertEqual(self.calc.memory_recall(), 8)

        self.calc.memory_subtract(2)
        self.assertEqual(self.calc.memory_recall(), 6)

        self.calc.memory_clear()
        self.assertEqual(self.calc.memory_recall(), 0)

if __name__ == '__main__':
    unittest.main()