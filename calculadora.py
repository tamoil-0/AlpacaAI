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
    
    def limpiar_historial(self):
        """Limpia el historial de operaciones"""
        self.historial = []
        return "Historial limpiado"
    
    def obtener_ultimo_resultado(self):
        """Retorna el último resultado calculado"""
        if not self.historial:
            return None
        return self.resultado
    
    def valor_absoluto(self, n):
        """Calcula el valor absoluto de un número"""
        resultado = abs(n)
        self.historial.append(f"|{n}| = {resultado}")
        return resultado
    
    def modulo(self, a, b):
        """Calcula el módulo (resto de la división)"""
        if b == 0:
            raise ValueError("No se puede calcular módulo con divisor cero")
        resultado = a % b
        self.historial.append(f"{a} mod {b} = {resultado}")
        return resultado
    
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
    
    def arcoseno(self, valor, grados=True):
        """Calcula el arcoseno (inversa del seno)"""
        if valor < -1 or valor > 1:
            raise ValueError("El valor debe estar entre -1 y 1")
        resultado = math.asin(valor)
        if grados:
            resultado = math.degrees(resultado)
        unidad = "°" if grados else "rad"
        self.historial.append(f"arcsin({valor}) = {resultado}{unidad}")
        return resultado
    
    def arcocoseno(self, valor, grados=True):
        """Calcula el arcocoseno (inversa del coseno)"""
        if valor < -1 or valor > 1:
            raise ValueError("El valor debe estar entre -1 y 1")
        resultado = math.acos(valor)
        if grados:
            resultado = math.degrees(resultado)
        unidad = "°" if grados else "rad"
        self.historial.append(f"arccos({valor}) = {resultado}{unidad}")
        return resultado
    
    def arcotangente(self, valor, grados=True):
        """Calcula la arcotangente (inversa de la tangente)"""
        resultado = math.atan(valor)
        if grados:
            resultado = math.degrees(resultado)
        unidad = "°" if grados else "rad"
        self.historial.append(f"arctan({valor}) = {resultado}{unidad}")
        return resultado
    
    def grados_a_radianes(self, grados):
        """Convierte grados a radianes"""
        resultado = math.radians(grados)
        self.historial.append(f"{grados}° = {resultado} rad")
        return resultado
    
    def radianes_a_grados(self, radianes):
        """Convierte radianes a grados"""
        resultado = math.degrees(radianes)
        self.historial.append(f"{radianes} rad = {resultado}°")
        return resultado
    
    def redondear(self, numero, decimales=0):
        """Redondea un número a un número específico de decimales"""
        resultado = round(numero, decimales)
        self.historial.append(f"round({numero}, {decimales}) = {resultado}")
        return resultado
    
    def signo(self, numero):
        """Devuelve el signo de un número: -1 si negativo, 0 si cero, 1 si positivo"""
        if numero > 0:
            resultado = 1
        elif numero < 0:
            resultado = -1
        else:
            resultado = 0
        self.historial.append(f"sign({numero}) = {resultado}")
        return resultado
    
    def minimo(self, a, b):
        """Devuelve el mínimo de dos números"""
        resultado = min(a, b)
        self.historial.append(f"min({a}, {b}) = {resultado}")
        return resultado
    
    def maximo(self, a, b):
        """Devuelve el máximo de dos números"""
        resultado = max(a, b)
        self.historial.append(f"max({a}, {b}) = {resultado}")
        return resultado
    
    def promedio(self, *numeros):
        """Calcula el promedio de una lista de números"""
        if not numeros:
            raise ValueError("Se requieren al menos un número para calcular el promedio")
        resultado = sum(numeros) / len(numeros)
        self.historial.append(f"avg({', '.join(map(str, numeros))}) = {resultado}")
        return resultado
    
    def celsius_a_fahrenheit(self, celsius):
        """Convierte grados Celsius a Fahrenheit"""
        resultado = (celsius * 9/5) + 32
        self.historial.append(f"{celsius}°C = {resultado}°F")
        return resultado
    
    def fahrenheit_a_celsius(self, fahrenheit):
        """Convierte grados Fahrenheit a Celsius"""
        resultado = (fahrenheit - 32) * 5/9
        self.historial.append(f"{fahrenheit}°F = {resultado}°C")
        return resultado
    
    def combinaciones(self, n, k):
        """Calcula el número de combinaciones de n elementos tomados de k en k"""
        if k > n or k < 0 or n < 0:
            raise ValueError("Valores inválidos para combinaciones")
        resultado = math.comb(n, k)
        self.historial.append(f"C({n}, {k}) = {resultado}")
        return resultado
    
    def permutaciones(self, n, k):
        """Calcula el número de permutaciones de n elementos tomados de k en k"""
        if k > n or k < 0 or n < 0:
            raise ValueError("Valores inválidos para permutaciones")
        resultado = math.perm(n, k)
        self.historial.append(f"P({n}, {k}) = {resultado}")
        return resultado
    
    def desviacion_estandar(self, *numeros):
        """Calcula la desviación estándar de una lista de números"""
        if len(numeros) < 2:
            raise ValueError("Se requieren al menos dos números para calcular la desviación estándar")
        import statistics
        resultado = statistics.stdev(numeros)
        self.historial.append(f"stdev({', '.join(map(str, numeros))}) = {resultado}")
        return resultado
    
    def varianza(self, *numeros):
        """Calcula la varianza de una lista de números"""
        if len(numeros) < 2:
            raise ValueError("Se requieren al menos dos números para calcular la varianza")
        import statistics
        resultado = statistics.variance(numeros)
        self.historial.append(f"var({', '.join(map(str, numeros))}) = {resultado}")
        return resultado
    
    def area_circulo(self, radio):
        """Calcula el área de un círculo dado su radio"""
        if radio < 0:
            raise ValueError("El radio no puede ser negativo")
        resultado = math.pi * radio ** 2
        self.historial.append(f"area_circle(r={radio}) = {resultado}")
        return resultado
    
    def mediana(self, *numeros):
        """Calcula la mediana de una lista de números"""
        if not numeros:
            raise ValueError("Se requieren números para calcular la mediana")
        import statistics
        resultado = statistics.median(numeros)
        self.historial.append(f"median({', '.join(map(str, numeros))}) = {resultado}")
        return resultado
    
    def logaritmo(self, n, base=10):
        """Calcula el logaritmo en base especificada"""
        if n <= 0:
            raise ValueError("No se puede calcular logaritmo de número no positivo")
        if base <= 0 or base == 1:
            raise ValueError("Base de logaritmo inválida")
        resultado = math.log(n, base)
        self.historial.append(f"log_{base}({n}) = {resultado}")
        return resultado
    
    def logaritmo_natural(self, n):
        """Calcula el logaritmo natural (ln)"""
        if n <= 0:
            raise ValueError("No se puede calcular logaritmo de número no positivo")
        resultado = math.log(n)
        self.historial.append(f"ln({n}) = {resultado}")
        return resultado
    
    def exponencial(self, n):
        """Calcula e^n"""
        resultado = math.exp(n)
        self.historial.append(f"e^{n} = {resultado}")
        return resultado
    
    def raiz_n(self, numero, indice):
        """Calcula la raíz enésima de un número"""
        if numero < 0 and indice % 2 == 0:
            raise ValueError("No se puede calcular raíz par de número negativo")
        if indice == 0:
            raise ValueError("Índice de raíz no puede ser cero")
        resultado = numero ** (1 / indice)
        self.historial.append(f"{indice}√{numero} = {resultado}")
        return resultado
    
    def porcentaje(self, cantidad, porcentaje):
        """Calcula el porcentaje de una cantidad"""
        resultado = (cantidad * porcentaje) / 100
        self.historial.append(f"{porcentaje}% de {cantidad} = {resultado}")
        return resultado
    
    def agregar_porcentaje(self, cantidad, porcentaje):
        """Agrega un porcentaje a una cantidad"""
        resultado = cantidad * (1 + porcentaje / 100)
        self.historial.append(f"{cantidad} + {porcentaje}% = {resultado}")
        return resultado
    
    def reducir_porcentaje(self, cantidad, porcentaje):
        """Reduce un porcentaje de una cantidad"""
        resultado = cantidad * (1 - porcentaje / 100)
        self.historial.append(f"{cantidad} - {porcentaje}% = {resultado}")
        return resultado
    
    def media(self, numeros):
        """Calcula la media aritmética de una lista de números"""
        if len(numeros) == 0:
            raise ValueError("La lista no puede estar vacía")
        resultado = sum(numeros) / len(numeros)
        self.historial.append(f"Media de {numeros} = {resultado}")
        return resultado
    
    def mediana(self, numeros):
        """Calcula la mediana de una lista de números"""
        if len(numeros) == 0:
            raise ValueError("La lista no puede estar vacía")
        ordenados = sorted(numeros)
        n = len(ordenados)
        if n % 2 == 0:
            resultado = (ordenados[n//2-1] + ordenados[n//2]) / 2
        else:
            resultado = ordenados[n//2]
        self.historial.append(f"Mediana de {numeros} = {resultado}")
        return resultado
    
    def desviacion_estandar(self, numeros):
        """Calcula la desviación estándar de una lista de números"""
        if len(numeros) == 0:
            raise ValueError("La lista no puede estar vacía")
        media = sum(numeros) / len(numeros)
        varianza = sum((x - media) ** 2 for x in numeros) / len(numeros)
        resultado = math.sqrt(varianza)
        self.historial.append(f"Desv. Est. de {numeros} = {resultado}")
        return resultado
    
    def convertir_longitud(self, valor, unidad_origen, unidad_destino):
        """Convierte unidades de longitud (m, km, cm, mm, ft, in, mi)"""
        # Factores a metros
        factores = {
            "m": 1, "km": 1000, "cm": 0.01, "mm": 0.001,
            "ft": 0.3048, "in": 0.0254, "mi": 1609.34
        }
        if unidad_origen not in factores or unidad_destino not in factores:
            raise ValueError("Unidad de longitud no soportada")
            
        valor_metros = valor * factores[unidad_origen]
        return valor_metros / factores[unidad_destino]
        
    def convertir_masa(self, valor, unidad_origen, unidad_destino):
        """Convierte unidades de masa (kg, g, mg, lb, oz)"""
        # Factores a kilogramos
        factores = {
            "kg": 1, "g": 0.001, "mg": 0.000001,
            "lb": 0.453592, "oz": 0.0283495
        }
        if unidad_origen not in factores or unidad_destino not in factores:
            raise ValueError("Unidad de masa no soportada")
            
        valor_kg = valor * factores[unidad_origen]
        return valor_kg / factores[unidad_destino]
        
    def convertir_temperatura(self, valor, unidad_origen, unidad_destino):
        """Convierte unidades de temperatura (C, F, K)"""
        if unidad_origen == unidad_destino:
            return valor
            
        # Convertir a Celsius primero
        if unidad_origen == "F":
            celsius = (valor - 32) * 5/9
        elif unidad_origen == "K":
            celsius = valor - 273.15
        else:
            celsius = valor
            
        # Convertir de Celsius a destino
        if unidad_destino == "F":
            return (celsius * 9/5) + 32
        elif unidad_destino == "K":
            return celsius + 273.15
        else:
            return celsius
            
    def resolver_ecuacion_cuadratica(self, a, b, c):
        """Resuelve ecuación ax^2 + bx + c = 0 usando fórmula general"""
        self.historial.append(f"Ec: {a}x² + {b}x + {c} = 0")
        
        if a == 0:
            if b == 0:
                if c == 0:
                    return "Infinitas soluciones"
                else:
                    return "Sin solución"
            else:
                x = -c / b
                return f"x = {x}"
        
        discriminante = b**2 - 4*a*c
        
        if discriminante > 0:
            x1 = (-b + math.sqrt(discriminante)) / (2*a)
            x2 = (-b - math.sqrt(discriminante)) / (2*a)
            return f"x₁ = {x1:.4f}\nx₂ = {x2:.4f}"
        elif discriminante == 0:
            x = -b / (2*a)
            return f"x = {x:.4f}"
        else:
            parte_real = -b / (2*a)
            parte_imag = math.sqrt(abs(discriminante)) / (2*a)
            return f"x₁ = {parte_real:.4f} + {parte_imag:.4f}i\nx₂ = {parte_real:.4f} - {parte_imag:.4f}i"


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





