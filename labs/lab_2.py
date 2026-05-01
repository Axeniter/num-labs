import numpy as np
import math
import matplotlib.pyplot as plt
from table_1 import get_function_data

# region Конечные разности

def build_finite_differences(y):
    """Построение таблицы конечных разностей"""
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i+1, j-1] - table[i, j-1]
    return table


def print_finite_differences(x, table):
    """Вывод таблицы конечных разностей"""
    n = len(x)
    header = f"{'x':<10} {'y':<12}"
    for j in range(1, n):
        header += f"{'Δ'+str(j)+'y':<12}"
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{x[i]:<10.4f} {table[i,0]:<12.6f}"
        for j in range(1, n - i):
            if abs(table[i,j]) < 1e-15:
                table[i,j] = 0.0
            row += f"{table[i,j]:<12.6f}"
        print(row)

# endregion

# region Интерполяционные формулы

def newton_forward(x_nodes, table, x_star, max_nodes=None):
    """1-я формула Ньютона (вперёд)"""
    n = len(x_nodes)
    h = x_nodes[1] - x_nodes[0]
    
    i0 = np.searchsorted(x_nodes, x_star) - 1
    i0 = max(0, min(i0, n - 2))
    
    if max_nodes is None:
        max_nodes = n - i0
    
    t = (x_star - x_nodes[i0]) / h
    result = table[i0, 0]
    product = 1.0
    
    for k in range(1, max_nodes):
        if i0 + k >= n:
            break
        product *= (t - (k - 1))
        result += product * table[i0, k] / math.factorial(k)
    
    used_idx = list(range(i0, min(i0 + max_nodes, n)))
    return result, used_idx


def newton_backward(x_nodes, table, x_star, max_nodes=None):
    """2-я формула Ньютона (назад)"""
    n = len(x_nodes)
    h = x_nodes[1] - x_nodes[0]
    
    i0 = np.searchsorted(x_nodes, x_star)
    i0 = max(1, min(i0, n - 1))
    
    if max_nodes is None:
        max_nodes = i0 + 1
    
    t = (x_star - x_nodes[i0]) / h
    result = table[i0, 0]
    product = 1.0
    
    for k in range(1, max_nodes):
        if i0 - k < 0:
            break
        product *= (t + (k - 1))
        result += product * table[i0 - k, k] / math.factorial(k)
    
    used_idx = list(range(i0, i0 - min(max_nodes, i0 + 1), -1))
    return result, used_idx


def gauss_forward(x_nodes, table, x_star, max_nodes=None):
    """1-я формула Гаусса"""
    n = len(x_nodes)
    h = x_nodes[1] - x_nodes[0]
    
    i0 = np.searchsorted(x_nodes, x_star) - 1
    i0 = max(0, min(i0, n - 2))
    
    if x_star - x_nodes[i0] > x_nodes[i0+1] - x_star:
        i0 = i0 + 1
    
    t = (x_star - x_nodes[i0]) / h
    
    if max_nodes is None:
        max_nodes = n
    
    result = table[i0, 0]
    used_idx = [i0]
    
    if i0 < n - 1 and 1 < max_nodes:
        result += t * table[i0, 1]
        used_idx.append(i0 + 1)
    
    if i0 > 0 and 2 < max_nodes:
        result += t * (t - 1) / 2 * table[i0 - 1, 2]
        used_idx.append(i0 - 1)
    
    if i0 > 0 and i0 + 1 < n - 1 and 3 < max_nodes:
        result += (t + 1) * t * (t - 1) / 6 * table[i0 - 1, 3]
        used_idx.append(i0 + 2)
    
    if i0 > 1 and 4 < max_nodes:
        result += (t + 1) * t * (t - 1) * (t - 2) / 24 * table[i0 - 2, 4]
        used_idx.append(i0 - 2)
    
    return result, used_idx


def gauss_backward(x_nodes, table, x_star, max_nodes=None):
    """2-я формула Гаусса"""
    n = len(x_nodes)
    h = x_nodes[1] - x_nodes[0]
    
    i0 = np.searchsorted(x_nodes, x_star) - 1
    i0 = max(0, min(i0, n - 2))
    
    if x_star - x_nodes[i0] > x_nodes[i0+1] - x_star:
        i0 = i0 + 1
    
    t = (x_star - x_nodes[i0]) / h
    
    if max_nodes is None:
        max_nodes = n
    
    result = table[i0, 0]
    used_idx = [i0]
    
    if i0 > 0 and 1 < max_nodes:
        result += t * table[i0 - 1, 1]
        used_idx.append(i0 - 1)
    
    if i0 > 0 and 2 < max_nodes:
        result += t * (t + 1) / 2 * table[i0 - 1, 2]
        used_idx.append(i0 + 1)
    
    if i0 > 1 and 3 < max_nodes:
        result += (t + 1) * t * (t - 1) / 6 * table[i0 - 2, 3]
        used_idx.append(i0 - 2)
    
    if i0 > 1 and 4 < max_nodes:
        result += (t + 1) * t * (t - 1) * (t + 2) / 24 * table[i0 - 2, 4]
        used_idx.append(i0 + 2)
    
    return result, used_idx


def stirling(x_nodes, table, x_star):
    """Формула Стирлинга: центральная, для t ≈ 0"""
    n = len(x_nodes)
    h = x_nodes[1] - x_nodes[0]
    
    i0 = np.searchsorted(x_nodes, x_star) - 1
    i0 = max(0, min(i0, n - 2))
    
    if abs(x_star - x_nodes[i0]) > abs(x_star - x_nodes[i0+1]):
        i0 = i0 + 1
    
    t = (x_star - x_nodes[i0]) / h
    
    result = table[i0, 0]
    used_idx = [i0]
    
    if i0 > 0 and i0 < n - 1:
        mu_delta = (table[i0, 1] + table[i0-1, 1]) / 2
    elif i0 == 0:
        mu_delta = table[0, 1]
    else:
        mu_delta = table[-2, 1]
    result += t * mu_delta
    if i0 > 0:
        used_idx.append(i0 - 1)
    if i0 < n - 1:
        used_idx.append(i0 + 1)
    
    if i0 > 0:
        result += t**2 / 2 * table[i0-1, 2]
    
    if i0 > 0 and i0 < n - 2:
        mu_delta3 = (table[i0-1, 3] + table[i0-2, 3]) / 2 if i0 > 1 else table[i0-1, 3]
        result += t * (t**2 - 1) / 6 * mu_delta3
    
    if i0 > 1:
        result += t**2 * (t**2 - 1) / 24 * table[i0-2, 4]
    
    return result, used_idx


def bessel(x_nodes, table, x_star):
    """Формула Бесселя"""
    n = len(x_nodes)
    h = x_nodes[1] - x_nodes[0]
    
    i0 = np.searchsorted(x_nodes, x_star) - 1
    i0 = max(0, min(i0, n - 2))
    
    t = (x_star - x_nodes[i0]) / h
    
    result = (table[i0, 0] + table[i0+1, 0]) / 2
    used_idx = [i0, i0+1]
    
    result += (t - 0.5) * table[i0, 1]
    
    if i0 > 0 and i0 < n - 2:
        mu_delta2 = (table[i0-1, 2] + table[i0, 2]) / 2
        result += t * (t - 1) / 2 * mu_delta2
        used_idx.append(i0-1)
        if i0 + 2 < n:
            used_idx.append(i0+2)
    
    if i0 > 0 and i0 + 1 < n - 1:
        result += t * (t - 1) * (t - 0.5) / 6 * table[i0-1, 3]
    
    if i0 > 1 and i0 + 2 < n:
        mu_delta4 = (table[i0-2, 4] + table[i0-1, 4]) / 2
        result += t * (t - 1) * (t + 1) * (t - 2) / 24 * mu_delta4
    
    return result, used_idx

# endregion

# region Выбор формулы

def select_best_formula(x_nodes, x_star):
    """Выбор наилучшей формулы"""
    i = np.searchsorted(x_nodes, x_star) - 1
    i = max(0, min(i, len(x_nodes) - 2))
    
    t = (x_star - x_nodes[i]) / (x_nodes[1] - x_nodes[0])
    n = len(x_nodes)
    
    if i < 2:
        return 'newton_forward', newton_forward
    elif i > n - 4:
        return 'newton_backward', newton_backward
    elif t < 0.25:
        return 'stirling', stirling
    elif t > 0.75:
        return 'bessel', bessel
    elif t <= 0.5:
        return 'gauss_forward', gauss_forward
    else:
        return 'gauss_backward', gauss_backward

# endregion

# region Оценка погрешности

def estimate_error_from_table(f_deriv, x_nodes, used_idx, x_star, order):
    """Оценка остаточного члена"""
    used_x = [x_nodes[k] for k in used_idx]
    omega = 1.0
    for xk in used_x:
        omega *= (x_star - xk)
    
    a, b = min(used_x), max(used_x)
    x_check = np.linspace(a, b, 100)
    
    try:
        deriv_vals = f_deriv(x_check)
        deriv_min = np.min(deriv_vals)
        deriv_max = np.max(deriv_vals)
    except:
        return 0, 0
    
    fact = math.factorial(order + 1)
    r_min = min(deriv_min * omega / fact, deriv_max * omega / fact)
    r_max = max(deriv_min * omega / fact, deriv_max * omega / fact)
    
    return r_min, r_max

# endregion

# region Решение

def solve(variant):
    data = get_function_data(variant)
    f = data['function']
    a, b = data['interval']
    x1, x2, x3, x4 = data['points']
    
    points = [x1, x2, x3, x4]
    labels = ['x*', 'x**', 'x***', 'x****']
    
    n_table = 11
    x_table = np.linspace(a, b, n_table)
    y_table = f(x_table)
    diff_table = build_finite_differences(y_table)
    
    print(f"\nВариант {variant}")
    print(f"Отрезок: [{a}, {b}]")
    print(f"{'='*50}")
    
    print_finite_differences(x_table, diff_table)
    
    for label, x_val in zip(labels, points):
        print(f"\nТочка {label} = {x_val:.6f}")
        print(f"{'-'*50}")
        
        f_exact = f(x_val)
        print(f"  f({label}) точное = {f_exact:.10f}")
        
        best_name, best_func = select_best_formula(x_table, x_val)
        print(f"  Рекомендуемая формула: {best_name}")
        
        L_val, used_idx = best_func(x_table, diff_table, x_val)
        error = abs(L_val - f_exact)
        
        print(f"  L({label}) = {L_val:.10f}")
        print(f"  Погрешность: {error:.2e}")
        print(f"{'-'*50}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x_plot = np.linspace(a, b, 400)
    f_plot = f(x_plot)
    
    errors_list = []
    methods_list = []
    
    ax = axes[0, 0]
    
    for label, x_val in zip(labels[1:], points[1:]):  # только 2, 3, 4
        best_name, best_func = select_best_formula(x_table, x_val)
        L_val, _ = best_func(x_table, diff_table, x_val)
        error = abs(L_val - f(x_val))
        errors_list.append(error)
        methods_list.append(f'{label}\n{best_name}')
    
    colors = ['orange', 'green', 'blue']
    bars = ax.bar(methods_list, errors_list, color=colors, alpha=0.7)
    ax.set_ylabel('Абсолютная погрешность')
    ax.set_title('Погрешности интерполяции')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, err in zip(bars, errors_list):
        ax.text(bar.get_x() + bar.get_width()/2, err * 2, f'{err:.2e}',
                ha='center', fontsize=9, fontweight='bold')
    
    for idx, (label, x_val) in enumerate(zip(labels[1:], points[1:])):
        ax = axes[(idx + 1) // 2, (idx + 1) % 2]
        
        margin = (b - a) * 0.12
        xl, xr = max(a, x_val - margin), min(b, x_val + margin)
        
        mask = (x_plot >= xl) & (x_plot <= xr)
        ax.plot(x_plot[mask], f_plot[mask], 'b-', lw=1.5, label='f(x)')
        
        mask_nodes = (x_table >= xl) & (x_table <= xr)
        ax.plot(x_table[mask_nodes], y_table[mask_nodes], 'ko', ms=6, label='узлы')
        
        ax.axvline(x_val, color='red', ls=':', alpha=0.5)
        ax.scatter([x_val], [f(x_val)], color='red', s=80, zorder=5)
        
        best_name, best_func = select_best_formula(x_table, x_val)
        x_interp = np.linspace(xl, xr, 200)
        y_interp = [best_func(x_table, diff_table, xi)[0] for xi in x_interp]
        ax.plot(x_interp, y_interp, 'r--', lw=1.5, label=best_name)
        
        ax.set_title(f'{label} = {x_val:.4f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xl, xr)
    
    plt.suptitle(f'Вариант {variant}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'lab_2_v{variant}.png', dpi=150)
    plt.show()

# region Запуск

if __name__ == "__main__":
    v = int(input("Введите номер варианта (1-24): "))
    solve(v)

# endregion