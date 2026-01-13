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
        print(Fore.RED + "  6. Salir")
        print(Fore.CYAN + "----------------------------")

        opcion = input(Fore.WHITE + "Elige una opción (1/2/3/4/5/6): ").strip()

        if opcion == '6':
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
            else:
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
        else:
            print(Fore.RED + "Opción no válida")
            input(Fore.WHITE + "Presiona Enter para continuar...")
            continue

        print(Fore.CYAN + "----------------------------")
        if isinstance(resultado, float):
            print(Fore.BLUE + f"Resultado: {resultado:.2f}")
        else:
            print(Fore.BLUE + f"Resultado: {resultado}")
        input(Fore.WHITE + "Presiona Enter para continuar...")


if __name__ == "__main__":
    main()