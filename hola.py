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
    print("============================")
    print("   Calculadora Básica   ")
    print("============================")
    print("Opciones:")
    print("  1. Suma")
    print("  2. Resta")
    print("  3. Multiplicación")
    print("  4. División")
    print("----------------------------")

    opcion = input("Elige una opción (1/2/3/4): ")

    a = float(input("Ingresa el primer número: "))
    b = float(input("Ingresa el segundo número: "))

    if opcion == '1':
        print("Resultado:", suma(a, b))
    elif opcion == '2':
        print("Resultado:", resta(a, b))
    elif opcion == '3':
        print("Resultado:", multiplicacion(a, b))
    elif opcion == '4':
        print("Resultado:", division(a, b))
    else:
        print("Opción no válida")

if __name__ == "__main__":
    main()