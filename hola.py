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
        print(Fore.RED + "  13. Salir")
        print(Fore.CYAN + "----------------------------")

        opcion = input(Fore.WHITE + "Elige una opción (1/2/3/4/5/6/7/8/9/10/11/12/13): ").strip()

        if opcion == '13':
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
            elif opcion in ['1','2','3','5','6','8','9','10']:
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
        elif isinstance(resultado, float):
            print(Fore.BLUE + f"Resultado: {resultado:.2f}")
        else:
            print(Fore.BLUE + f"Resultado: {resultado}")
        input(Fore.WHITE + "Presiona Enter para continuar...")


if __name__ == "__main__":
    main()