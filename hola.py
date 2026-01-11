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
    print(Fore.CYAN + "============================")
    print(Fore.YELLOW + "   Calculadora Básica   ")
    print(Fore.CYAN + "============================")
    print(Fore.GREEN + "Opciones:")
    print(Fore.MAGENTA + "  1. Suma")
    print(Fore.MAGENTA + "  2. Resta")
    print(Fore.MAGENTA + "  3. Multiplicación")
    print(Fore.MAGENTA + "  4. División")
    print(Fore.CYAN + "----------------------------")

    opcion = input(Fore.WHITE + "Elige una opción (1/2/3/4): ").strip()

    try:
        a = float(input(Fore.WHITE + "Ingresa el primer número: "))
        b = float(input(Fore.WHITE + "Ingresa el segundo número: "))
    except ValueError:
        print(Fore.RED + "Error: Ingresa solo números válidos.")
        return

    if opcion == '1':
        print(Fore.BLUE + "Resultado:", suma(a, b))
    elif opcion == '2':
        print(Fore.BLUE + "Resultado:", resta(a, b))
    elif opcion == '3':
        print(Fore.BLUE + "Resultado:", multiplicacion(a, b))
    elif opcion == '4':
        print(Fore.BLUE + "Resultado:", division(a, b))
    else:
        print(Fore.RED + "Opción no válida")

if __name__ == "__main__":
    main()