
import numpy as np

def gauss_seidel(A, b, x0, tolerance, max_iterations):
    n = len(A)
    x = x0.copy()
    
    for k in range(max_iterations):
        x_old = x.copy()
        
        for i in range(n):
            # Suma de los elementos de la fila i excepto el diagonal
            sum1 = sum(A[i][j] * x[j] for j in range(n) if j != i)
            
            # Actualizar x_i
            x[i] = (b[i] - sum1) / A[i][i]
        
        # Comprobar la convergencia
        if np.linalg.norm(np.array(x) - np.array(x_old), ord=np.inf) < tolerance:
            return x, k + 1  # Retorna la solucion y el numero de iteraciones
    
    return x, max_iterations  # Retorna la solucion y el numero maximo de iteraciones alcanzado

# Definir la matriz A y el vector b
A = np.array([[0.52, 0.3, 0.18],
              [0.2, 0.5, 0.3],
              [0.25, 0.2, 0.55]])

b = np.array([4800, 5810, 5690])

# Valores iniciales
x0 = [0, 0, 0]  # Estimacion inicial para cada variable
tolerance = 1e-6
max_iterations = 100

# Ejecutar el metodo de Gauss-Seidel
solution, iterations = gauss_seidel(A, b, x0, tolerance, max_iterations)

# Imprimir los resultados
print("Solucion:", solution)
print("Iteraciones:", iterations)
