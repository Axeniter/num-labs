import numpy as np
from typing import Tuple


# region Метод монотонной прогонки

def tridiagonal_method(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    F: np.ndarray
) -> np.ndarray:
    """
    Решение трёхдиагональной системы методом правой монотонной прогонки

    Параметры:
    - A: массив коэффициентов нижней диагонали
    - B: массив коэффициентов главной диагонали
    - C: массив коэффициентов верхней диагонали
    - F: массив правых частей
    
    Возвращает:
    - U: массив решения
    """
    N = len(F)
    
    alpha = np.zeros(N)
    beta = np.zeros(N)
    
    alpha[0] = -C[0] / B[0]
    beta[0] = F[0] / B[0]
    
    for k in range(1, N - 1):
        denominator = B[k] + A[k] * alpha[k - 1]
        
        alpha[k] = -C[k] / denominator
        beta[k] = (F[k] - A[k] * beta[k - 1]) / denominator
    
    U = np.zeros(N)
    last_denom = A[-1] * alpha[-2] + B[-1]
    U[-1] = (F[-1] - A[-1] * beta[-2]) / last_denom
    
    for k in range(N - 2, -1, -1):
        U[k] = alpha[k] * U[k + 1] + beta[k]
    
    return U


def tridiagonal_solve(
    matrix: np.ndarray,
    f: np.ndarray
) -> np.ndarray:
    """Решение трёхдиагональной системы"""
    N = len(f)

    A = np.zeros(N)
    B = np.zeros(N)
    C = np.zeros(N)
    
    for i in range(N):
        B[i] = matrix[i, i]
        if i > 0:
            A[i] = matrix[i, i-1]
        if i < N - 1:
            C[i] = matrix[i, i+1]
    
    return tridiagonal_method(A, B, C, f)

# endregion

# region Генерация трехдиагональной системы

def generate_tridiagonal_system(
    n: int,
    d: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Генерирует случайную трёхдиагональную систему"""
    matrix = np.zeros((n, n))
    
    np.fill_diagonal(matrix, np.random.uniform(-d, d, n))
    
    lower = np.random.uniform(-d, d, n - 1)
    np.fill_diagonal(matrix[1:], lower)
    
    upper = np.random.uniform(-d, d, n - 1)
    np.fill_diagonal(matrix[:, 1:], upper)
    
    f = np.random.uniform(-d, d, n)
    
    return matrix, f

# endregion

# region Решение системы

def check_residual(A, b, x):
    """Проверка невязки решения"""
    return np.linalg.norm(A @ x - b)

def main():
    while True:
        n = int(input("Введите размерность системы (n): "))
        d = float(input("Введите разброс значений (-d, d): "))

        A, b = generate_tridiagonal_system(n, d)
        print("\nМатрица:")
        print(A)
        print("\nВектор:")
        print(b)

        x = tridiagonal_solve(A, b)
        print("\nРешение:")
        print(x)

        r = check_residual(A, b, x)
        print("\nНевязка:")
        print(r)

        input("\nПродолжить...")

# endregion

# region Запуск

if __name__ == "__main__":
    main()

# endregion