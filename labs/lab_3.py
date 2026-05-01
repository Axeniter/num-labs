import numpy as np
import matplotlib.pyplot as plt
import math
from table_1 import get_function_data
from table_2 import get_variant_params

# region Дифференцирование

def lagrange_basis_derivative(i, m, n, k):
    if k == 1:
        deriv = 0.0
        for s in range(n + 1):
            if s == i:
                continue
            term = 1.0 / (i - s)
            for j in range(n + 1):
                if j == i or j == s:
                    continue
                term *= (m - j) / (i - j)
            deriv += term
        return deriv
    
    elif k == 2:
        deriv = 0.0
        for s in range(n + 1):
            if s == i:
                continue
            for r in range(n + 1):
                if r == i or r == s:
                    continue
                term = 2.0 / ((i - s) * (i - r))
                for j in range(n + 1):
                    if j == i or j == s or j == r:
                        continue
                    term *= (m - j) / (i - j)
                deriv += term
        return deriv
    
    return 0.0


def omega_derivative(m, n, k, h):
    if k == 1:
        result = 0.0
        for s in range(n + 1):
            term = 1.0
            for j in range(n + 1):
                if j == s:
                    continue
                term *= (m - j)
            result += term
        return result * h ** (n + 1 - k)
    
    elif k == 2:
        result = 0.0
        for s in range(n + 1):
            for r in range(n + 1):
                if r == s:
                    continue
                term = 1.0
                for j in range(n + 1):
                    if j == s or j == r:
                        continue
                    term *= (m - j)
                result += term
        return result * h ** (n + 1 - k)
    
    return 0.0

# endregion

# region Решение

def solve_numerical_derivative(func_var, variant):
    params = get_variant_params(variant)
    k = params['k']
    n = params['n']
    m = params['m']
    
    data = get_function_data(func_var)
    f = data['function']
    f_deriv_exact = data['first_derivative'] if k == 1 else data['second_derivative']
    a, b = data['interval']
    
    if n + 1 == 1:
        f_high = data['first_derivative']
    elif n + 1 == 2:
        f_high = data['second_derivative']
    elif n + 1 == 3:
        f_high = data['third_derivative']
    else:
        f_high = data['third_derivative']
    
    h = (b - a) / n
    x_nodes = np.array([a + i * h for i in range(n + 1)])
    y_nodes = f(x_nodes)
    x_m = x_nodes[m]
    
    coeffs = np.zeros(n + 1)
    for i in range(n + 1):
        coeffs[i] = lagrange_basis_derivative(i, m, n, k)
    
    approx = np.sum(coeffs * y_nodes) / (h ** k)
    exact = f_deriv_exact(x_m)
    error = approx - exact
    
    omega_k = omega_derivative(m, n, k, h)
    
    x_check = np.linspace(a, b, 200)
    f_high_vals = f_high(x_check)
    M_min = np.min(f_high_vals)
    M_max = np.max(f_high_vals)
    
    fact = math.factorial(n + 1)
    R_min = min(M_min * omega_k / fact, M_max * omega_k / fact)
    R_max = max(M_min * omega_k / fact, M_max * omega_k / fact)
    
    print(f"\nЧисленное дифференцирование")
    print(f"{'='*50}")
    print(f"Функция: вариант {func_var}")
    print(f"k={k}, n={n}, m={m}")
    print(f"Интервал: [{a}, {b}], h={h:.6f}")
    print(f"x_m = {x_m:.6f}")
    print(f"{'-'*50}")
    print(f"Приближённо: f^({k})({x_m:.4f}) = {approx:.10f}")
    print(f"Точно:       f^({k})({x_m:.4f}) = {exact:.10f}")
    print(f"Погрешность: {abs(error):.2e}")
    print(f"Оценка R:    [{R_min:.2e}, {R_max:.2e}]")
    print(f"Проверка:    {R_min:.2e} ≤ {error:.2e} ≤ {R_max:.2e}")
    print(f"{'-'*50}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x_plot = np.linspace(a, b, 300)
    
    ax1.plot(x_plot, f_deriv_exact(x_plot), 'b-', lw=2, alpha=0.6, label=f'f^({k})(x) точная')
    ax1.plot(x_nodes, f_deriv_exact(x_nodes), 'bo', ms=6, label='узлы точной')
    
    L_deriv_vals = np.zeros_like(x_plot)
    for idx, xi in enumerate(x_plot):
        coeffs_i = np.zeros(n + 1)
        for i in range(n + 1):
            t = (xi - a) / h
            if k == 1:
                val = 0.0
                for s in range(n + 1):
                    if s == i: continue
                    term = 1.0 / (i - s)
                    for j in range(n + 1):
                        if j == i or j == s: continue
                        term *= (t - j) / (i - j)
                    val += term
            else:
                val = 0.0
                for s in range(n + 1):
                    if s == i: continue
                    for r in range(n + 1):
                        if r == i or r == s: continue
                        term = 2.0 / ((i - s) * (i - r))
                        for j in range(n + 1):
                            if j == i or j == s or j == r: continue
                            term *= (t - j) / (i - j)
                        val += term
            coeffs_i[i] = val
        L_deriv_vals[idx] = np.sum(coeffs_i * y_nodes) / (h ** k)
    
    ax1.plot(x_plot, L_deriv_vals, 'r--', lw=1.5, label=f'L_{n}^({k})(x) приближенная')
    ax1.axvline(x_m, color='gray', ls=':', alpha=0.5)
    ax1.scatter([x_m], [approx], color='red', s=100, zorder=5, marker='*', label=f'x_{m}')
    ax1.set_title(f'Производная порядка {k}')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    errors_nodes = []
    for i in range(n + 1):
        coeffs_i = np.array([lagrange_basis_derivative(ii, i, n, k) for ii in range(n + 1)])
        L_val = np.sum(coeffs_i * y_nodes) / (h ** k)
        errors_nodes.append(abs(L_val - f_deriv_exact(x_nodes[i])))
    
    bars = ax2.bar(range(n + 1), errors_nodes, color='steelblue', alpha=0.7)
    ax2.bar(m, errors_nodes[m], color='red', alpha=0.8)
    ax2.set_title(f'|L_n^({k})(x_i) - f^({k})(x_i)| в узлах')
    ax2.set_xticks(range(n + 1))
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, err) in enumerate(zip(bars, errors_nodes)):
        ax2.text(bar.get_x() + bar.get_width()/2, err + max(errors_nodes)*0.02,
                 f'{err:.2e}', ha='center', fontsize=8)
    
    plt.suptitle(f'Численное дифференцирование (func={func_var}, k={k}, n={n}, m={m})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'lab_3_v{variant}_f{func_var}.png', dpi=150)
    plt.show()

# endregion

# region Запуск

if __name__ == "__main__":
    func_var = int(input("Вариант функции (1-24): "))
    var = int(input("Вариант задания (1-37): "))
    solve_numerical_derivative(func_var, var)

# endregion