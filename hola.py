def grados_a_radianes(a):
    """Convierte grados a radianes."""
    return math.radians(a)

def radianes_a_grados(a):
    """Convierte radianes a grados."""
    return math.degrees(a)

def redondear(a, decimales=0):
    """Redondea un número a un número específico de decimales."""
    return round(a, decimales)

def signo(a):
    """Devuelve el signo de un número: -1 si negativo, 0 si cero, 1 si positivo."""
    if a > 0:
        return 1
    elif a < 0:
        return -1
    else:
        return 0

def mcd(a, b):
    """Devuelve el máximo común divisor de dos números enteros."""
    if not (float(a).is_integer() and float(b).is_integer()):
        return "Error: MCD solo está definido para enteros."
    return math.gcd(int(a), int(b))

def mcm(a, b):
    """Devuelve el mínimo común múltiplo de dos números enteros."""
    if not (float(a).is_integer() and float(b).is_integer()):
        return "Error: MCM solo está definido para enteros."
    return math.lcm(int(a), int(b))

def piso(a):
    """Devuelve el piso de un número (el entero más grande menor o igual a a)."""
    return math.floor(a)

def techo(a):
    """Devuelve el techo de un número (el entero más pequeño mayor o igual a a)."""
    return math.ceil(a)

def combinaciones(n, k):
    """Devuelve el número de combinaciones de n elementos tomados de k en k."""
    if not (float(n).is_integer() and float(k).is_integer()) or n < 0 or k < 0 or k > n:
        return "Error: n y k deben ser enteros no negativos con k <= n."
    return math.comb(int(n), int(k))

def permutaciones(n, k):
    """Devuelve el número de permutaciones de n elementos tomados de k en k."""
    if not (float(n).is_integer() and float(k).is_integer()) or n < 0 or k < 0 or k > n:
        return "Error: n y k deben ser enteros no negativos con k <= n."
    return math.perm(int(n), int(k))
def tangente(a):
    """Devuelve la tangente de un ángulo en radianes."""
    return math.tan(a)
def coseno(a):
    """Devuelve el coseno de un ángulo en radianes."""
    return math.cos(a)
def seno(a):
    """Devuelve el seno de un ángulo en radianes."""
    return math.sin(a)
def exponente(a):
    """Devuelve e elevado a la a."""
    return math.exp(a)
def log_base_10(a):
    """Devuelve el logaritmo en base 10 de un número positivo."""
    if a <= 0:
        return "Error: El logaritmo solo está definido para números positivos."
    return math.log10(a)
def log_natural(a):
    """Devuelve el logaritmo natural (base e) de un número positivo."""
    if a <= 0:
        return "Error: El logaritmo solo está definido para números positivos."
    return math.log(a)
def factorial(a):
    """Devuelve el factorial de un número entero no negativo."""
    if a < 0 or not float(a).is_integer():
        return "Error: El factorial solo está definido para enteros no negativos."
    return math.factorial(int(a))
def modulo(a, b):
    """Devuelve el módulo (resto) de la división de a entre b."""
    if b == 0:
        return "Error: No se puede dividir por cero."
    return a % b
def inverso(a):
    """Devuelve el inverso de un número (1/x)."""
    if a == 0:
        return "Error: No se puede dividir por cero."
    return 1 / a
import math
def raiz_cuadrada(a):
    """Devuelve la raíz cuadrada de un número."""
    if a < 0:
        return "Error: No se puede calcular la raíz cuadrada de un número negativo."
    return math.sqrt(a)
def potencia(a, b):
    """Devuelve a elevado a la potencia b."""
    return a ** b
def maximo(a, b):
    """Devuelve el mayor de dos números."""
    return max(a, b)
def minimo(a, b):
    """Devuelve el menor de dos números."""
    return min(a, b)
def valor_absoluto(a):
    """Devuelve el valor absoluto de un número."""
    return abs(a)
def promedio(a, b):
    """Devuelve el promedio de dos números."""
    return (a + b) / 2
def limpiar_ultima_linea():
    """Simula limpiar la última línea de la consola."""
    print("\033[F\033[K", end="")
def porcentaje(a, b):
    """Devuelve qué porcentaje es a de b."""
    if b == 0:
        return "Error: División por cero"
    return (a / b) * 100
import time
"""
Calculadora básica en consola
Autor: Jhon (mejoras automáticas)
Fecha: 2026-01-10
"""
import os
from colorama import init, Fore, Style

def suma(a, b):
    """Suma dos números y devuelve el resultado."""
    return a + b

def resta(a, b):
    """Resta el segundo número al primero y devuelve el resultado."""
    return a - b

def multiplicacion(a, b):
    """Multiplica dos números y devuelve el resultado."""
    return a * b

def division(a, b):
    """Divide el primer número por el segundo y devuelve el resultado."""
    if b == 0:
        return "Error: División por cero"
    return a / b

def main():
    init(autoreset=True)
    while True:
        limpiar_pantalla()
        print(Fore.CYAN + "============================")
        print(Fore.YELLOW + "   Calculadora Básica   ")
        print(Fore.CYAN + "============================")
        print(Fore.GREEN + "Opciones:")
        print(Fore.MAGENTA + "  1. Suma")
        print(Fore.MAGENTA + "  2. Resta")
        print(Fore.MAGENTA + "  3. Multiplicación")
        print(Fore.MAGENTA + "  4. División")
        print(Fore.MAGENTA + "  5. Porcentaje")
        print(Fore.MAGENTA + "  6. Promedio")
        print(Fore.MAGENTA + "  7. Valor absoluto")
        print(Fore.MAGENTA + "  8. Mínimo")
        print(Fore.MAGENTA + "  9. Máximo")
        print(Fore.MAGENTA + "  10. Potencia")
        print(Fore.MAGENTA + "  11. Raíz cuadrada")
        print(Fore.MAGENTA + "  12. Inverso")
        print(Fore.MAGENTA + "  13. Módulo")
        print(Fore.MAGENTA + "  14. Factorial")
        print(Fore.MAGENTA + "  15. Logaritmo natural")
        print(Fore.MAGENTA + "  16. Logaritmo base 10")
        print(Fore.MAGENTA + "  17. Exponente (e^x)")
        print(Fore.MAGENTA + "  18. Seno (radianes)")
        print(Fore.MAGENTA + "  19. Coseno (radianes)")
        print(Fore.MAGENTA + "  20. Tangente (radianes)")
        print(Fore.MAGENTA + "  21. Grados a radianes")
        print(Fore.MAGENTA + "  22. Radianes a grados")
        print(Fore.MAGENTA + "  23. Redondear")
        print(Fore.MAGENTA + "  24. Signo")
        print(Fore.MAGENTA + "  25. MCD")
        print(Fore.MAGENTA + "  26. MCM")
        print(Fore.MAGENTA + "  27. Piso")
        print(Fore.MAGENTA + "  28. Techo")
        print(Fore.MAGENTA + "  29. Combinaciones")
        print(Fore.MAGENTA + "  30. Permutaciones")
        print(Fore.RED + "  31. Salir")
        print(Fore.CYAN + "----------------------------")

        opcion = input(Fore.WHITE + "Elige una opción (1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31): ").strip()

        if opcion == '31':
            print(Fore.YELLOW + "¡Hasta luego!")
            time.sleep(1)
            break

        try:
            a = float(input(Fore.WHITE + "Ingresa el primer número: "))
            if opcion == '4':
                b = float(input(Fore.WHITE + "Ingresa el segundo número (no puede ser 0): "))
                if b == 0:
                    print(Fore.RED + "Error: No se puede dividir por cero.")
                    input(Fore.WHITE + "Presiona Enter para continuar...")
                    continue
            elif opcion in ['1','2','3','5','6','8','9','10','13','23','25','26','29','30']:
                b = float(input(Fore.WHITE + "Ingresa el segundo número: "))
        except ValueError:
            print(Fore.RED + "Error: Ingresa solo números válidos.")
            input(Fore.WHITE + "Presiona Enter para continuar...")
            continue

        if opcion == '1':
            resultado = suma(a, b)
        elif opcion == '2':
            resultado = resta(a, b)
        elif opcion == '3':
            resultado = multiplicacion(a, b)
        elif opcion == '4':
            resultado = division(a, b)
        elif opcion == '5':
            resultado = porcentaje(a, b)
        elif opcion == '6':
            resultado = promedio(a, b)
        elif opcion == '7':
            resultado = valor_absoluto(a)
        elif opcion == '8':
            resultado = minimo(a, b)
        elif opcion == '9':
            resultado = maximo(a, b)
        elif opcion == '10':
            resultado = potencia(a, b)
        elif opcion == '11':
            resultado = raiz_cuadrada(a)
        elif opcion == '12':
            resultado = inverso(a)
        elif opcion == '13':
            resultado = modulo(a, b)
        elif opcion == '14':
            resultado = factorial(a)
        elif opcion == '15':
            resultado = log_natural(a)
        elif opcion == '16':
            resultado = log_base_10(a)
        elif opcion == '17':
            resultado = exponente(a)
        elif opcion == '18':
            resultado = seno(a)
        elif opcion == '19':
            resultado = coseno(a)
        elif opcion == '20':
            resultado = tangente(a)
        elif opcion == '21':
            resultado = grados_a_radianes(a)
        elif opcion == '22':
            resultado = radianes_a_grados(a)
        elif opcion == '23':
            resultado = redondear(a, int(b))
        elif opcion == '24':
            resultado = signo(a)
        elif opcion == '25':
            resultado = mcd(a, b)
        elif opcion == '26':
            resultado = mcm(a, b)
        elif opcion == '27':
            resultado = piso(a)
        elif opcion == '28':
            resultado = techo(a)
        elif opcion == '29':
            resultado = combinaciones(a, b)
        elif opcion == '30':
            resultado = permutaciones(a, b)
        else:
            print(Fore.RED + "Opción no válida")
            input(Fore.WHITE + "Presiona Enter para continuar...")
            continue

        print(Fore.CYAN + "----------------------------")
        if opcion == '5' and isinstance(resultado, float):
            print(Fore.BLUE + f"Resultado: {resultado:.2f}%")
        elif opcion == '6' and isinstance(resultado, float):
            print(Fore.BLUE + f"Promedio: {resultado:.2f}")
        elif opcion == '7':
            print(Fore.BLUE + f"Valor absoluto: {resultado}")
        elif opcion == '8':
            print(Fore.BLUE + f"Mínimo: {resultado}")
        elif opcion == '9':
            print(Fore.BLUE + f"Máximo: {resultado}")
        elif opcion == '11':
            print(Fore.BLUE + f"Raíz cuadrada: {resultado}")
        elif opcion == '12':
            print(Fore.BLUE + f"Inverso: {resultado}")
        elif opcion == '14':
            print(Fore.BLUE + f"Factorial: {resultado}")
        elif opcion == '15':
            print(Fore.BLUE + f"Logaritmo natural: {resultado}")
        elif opcion == '16':
            print(Fore.BLUE + f"Logaritmo base 10: {resultado}")
        elif opcion == '18':
            print(Fore.BLUE + f"Seno: {resultado}")
        elif opcion == '19':
            print(Fore.BLUE + f"Coseno: {resultado}")
        elif opcion == '20':
            print(Fore.BLUE + f"Tangente: {resultado}")
        elif opcion == '21':
            print(Fore.BLUE + f"Radianes: {resultado}")
        elif opcion == '22':
            print(Fore.BLUE + f"Grados: {resultado}")
        elif opcion == '24':
            print(Fore.BLUE + f"Signo: {resultado}")
        elif opcion == '25':
            print(Fore.BLUE + f"MCD: {resultado}")
        elif opcion == '26':
            print(Fore.BLUE + f"MCM: {resultado}")
        elif opcion == '27':
            print(Fore.BLUE + f"Piso: {resultado}")
        elif opcion == '28':
            print(Fore.BLUE + f"Techo: {resultado}")
        elif opcion == '29':
            print(Fore.BLUE + f"Combinaciones: {resultado}")
        elif opcion == '30':
            print(Fore.BLUE + f"Permutaciones: {resultado}")
        elif isinstance(resultado, float):
            print(Fore.BLUE + f"Resultado: {resultado:.2f}")
        else:
            print(Fore.BLUE + f"Resultado: {resultado}")
        input(Fore.WHITE + "Presiona Enter para continuar...")


if __name__ == "__main__":
    main()