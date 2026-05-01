import numpy as np
import matplotlib.pyplot as plt
from table_1 import get_function_data

# region Интерполяция Лагранжа

def lagrange_linear(x_nodes, y_nodes, x_star):
    """Линейная формула Лагранжа"""
    x0, x1 = x_nodes
    y0, y1 = y_nodes
    return y0 * (x_star - x1) / (x0 - x1) + y1 * (x_star - x0) / (x1 - x0)


def lagrange_quadratic(x_nodes, y_nodes, x_star):
    """Квадратичная формула Лагранжа"""
    x0, x1, x2 = x_nodes
    y0, y1, y2 = y_nodes
    return (y0 * (x_star - x1) * (x_star - x2) / ((x0 - x1) * (x0 - x2)) +
            y1 * (x_star - x0) * (x_star - x2) / ((x1 - x0) * (x1 - x2)) +
            y2 * (x_star - x0) * (x_star - x1) / ((x2 - x0) * (x2 - x1)))


def lagrange_polynomial(x_nodes, y_nodes, x):
    """Многочлен Лагранжа произвольной степени"""
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        term = y_nodes[i]
        for j in range(n):
            if i != j:
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result

# endregion

# region Интерполяция Ньютона

def divided_differences(x_nodes, y_nodes):
    """Таблица разделенных разностей"""
    n = len(x_nodes)
    table = np.zeros((n, n))
    table[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i+1, j-1] - table[i, j-1]) / (x_nodes[i+j] - x_nodes[i])
    return table


def newton_polynomial(x_nodes, y_nodes, x):
    """Многочлен Ньютона с разделёнными разностями"""
    table = divided_differences(x_nodes, y_nodes)
    n = len(x_nodes)
    result = table[0, 0]
    product = 1.0
    for i in range(1, n):
        product *= (x - x_nodes[i-1])
        result += table[0, i] * product
    return result

# endregion

# region Оценка погрешности

def estimate_linear_error(f_second_deriv, x_nodes, x_star):
    """Оценка погрешности линейной интерполяции"""
    x0, x1 = x_nodes
    omega = (x_star - x0) * (x_star - x1)
    x_check = np.linspace(x0, x1, 100)
    fpp_vals = f_second_deriv(x_check)
    fpp_min, fpp_max = np.min(fpp_vals), np.max(fpp_vals)
    r_min = min(fpp_min * omega / 2, fpp_max * omega / 2)
    r_max = max(fpp_min * omega / 2, fpp_max * omega / 2)
    return r_min, r_max


def estimate_quadratic_error(f_third_deriv, x_nodes, x_star):
    """Оценка погрешности квадратичной интерполяции"""
    x0, x1, x2 = x_nodes
    omega = (x_star - x0) * (x_star - x1) * (x_star - x2)
    x_check = np.linspace(x0, x2, 100)
    fppp_vals = f_third_deriv(x_check)
    fppp_min, fppp_max = np.min(fppp_vals), np.max(fppp_vals)
    r_min = min(fppp_min * omega / 6, fppp_max * omega / 6)
    r_max = max(fppp_min * omega / 6, fppp_max * omega / 6)
    return r_min, r_max

# endregion

# region Решение

def solve_interpolation(variant):
    """Решение задачи интерполяции для выбранного варианта"""
    
    data = get_function_data(variant)
    f = data['function']
    f_second = data['second_derivative']
    f_third = data['third_derivative']
    a, b = data['interval']
    x_star, x2, x3, x4 = data['points']
    
    n_table = 11
    x_table = np.linspace(a, b, n_table)
    y_table = f(x_table)
    
    print(f"\n{'='*60}")
    print(f"ВАРИАНТ {variant}")
    print(f"{'='*60}")
    print(f"Отрезок: [{a}, {b}]")
    print(f"Точка интерполяции x* = {x_star}")
    
    f_exact = f(x_star)
    
    i = np.searchsorted(x_table, x_star) - 1
    i = max(0, min(i, n_table - 2))
    
    x_linear = np.array([x_table[i], x_table[i+1]])
    y_linear = f(x_linear)
    
    L1_lagrange = lagrange_linear(x_linear, y_linear, x_star)
    L1_newton = newton_polynomial(x_linear, y_linear, x_star)
    R1_actual = L1_lagrange - f_exact
    r1_min, r1_max = estimate_linear_error(f_second, x_linear, x_star)
    
    print(f"\nЛинейная интерполяция")
    print(f"Узлы: [{x_linear[0]:.6f}, {x_linear[1]:.6f}]")
    print(f"L1 Лагранж: {L1_lagrange:.10f}")
    print(f"L1 Ньютон:  {L1_newton:.10f}")
    print(f"f(x*) точн: {f_exact:.10f}")
    print(f"R1 фактич:  {R1_actual:.2e}")
    print(f"R1 оценка:  [{r1_min:.2e}, {r1_max:.2e}]")
    print(f"|R1| < 1e-4: {'да' if abs(R1_actual) <= 1e-4 else 'нет'}")
    
    if i == 0:
        idx = 0
    elif i >= n_table - 2:
        idx = n_table - 3
    elif x_star - x_table[i] < x_table[i+1] - x_star:
        idx = i - 1
    else:
        idx = i
    idx = max(0, min(idx, n_table - 3))
    
    x_quad = np.array([x_table[idx], x_table[idx+1], x_table[idx+2]])
    y_quad = f(x_quad)
    
    L2_lagrange = lagrange_quadratic(x_quad, y_quad, x_star)
    L2_newton = newton_polynomial(x_quad, y_quad, x_star)
    R2_actual = L2_lagrange - f_exact
    r2_min, r2_max = estimate_quadratic_error(f_third, x_quad, x_star)
    
    print(f"\nКвадратичная интерполяция")
    print(f"Узлы: [{x_quad[0]:.6f}, {x_quad[1]:.6f}, {x_quad[2]:.6f}]")
    print(f"L2 Лагранж: {L2_lagrange:.10f}")
    print(f"L2 Ньютон:  {L2_newton:.10f}")
    print(f"f(x*) точн: {f_exact:.10f}")
    print(f"R2 фактич:  {R2_actual:.2e}")
    print(f"R2 оценка:  [{r2_min:.2e}, {r2_max:.2e}]")
    print(f"|R2| < 1e-5: {'да' if abs(R2_actual) <= 1e-5 else 'нет'}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    margin = (b - a) * 0.15
    x_left = max(a, x_star - margin)
    x_right = min(b, x_star + margin)
    
    x_plot = np.linspace(x_left, x_right, 400)
    f_plot = f(x_plot)
    
    mask = (x_table >= x_left) & (x_table <= x_right)
    x_table_vis = x_table[mask]
    y_table_vis = y_table[mask]
    
    ax = axes[0]
    ax.plot(x_plot, f_plot, 'b-', linewidth=1.5, label='f(x) точная')
    ax.plot(x_table_vis, y_table_vis, 'ko', markersize=6, label='узлы таблицы')
    
    x_lin_plot = np.linspace(x_linear[0], x_linear[1], 50)
    L1_plot = [lagrange_linear(x_linear, y_linear, x) for x in x_lin_plot]
    ax.plot(x_lin_plot, L1_plot, 'r--', linewidth=1.5, label='L1 (линейный)')
    
    x_quad_plot = np.linspace(x_quad[0], x_quad[2], 80)
    L2_plot = [lagrange_quadratic(x_quad, y_quad, x) for x in x_quad_plot]
    ax.plot(x_quad_plot, L2_plot, 'g--', linewidth=1.5, label='L2 (квадратичный)')
    
    ax.axvline(x_star, color='red', linestyle=':', alpha=0.5)
    ax.scatter([x_star], [f_exact], color='red', s=80, zorder=5, label=f'x* = {x_star:.4f}')
    
    ax.set_title(f'Вариант {variant}: интерполяция в окрестности x*')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_left, x_right)
    
    ax = axes[1]
    methods = ['L1 Лагранж', 'L1 Ньютон', 'L2 Лагранж', 'L2 Ньютон']
    errors = [
        abs(L1_lagrange - f_exact),
        abs(L1_newton - f_exact),
        abs(L2_lagrange - f_exact),
        abs(L2_newton - f_exact)
    ]
    colors = ['red', 'orange', 'green', 'lime']
    bars = ax.bar(methods, errors, color=colors, alpha=0.7)
    ax.set_ylabel('Абсолютная погрешность')
    ax.set_title(f'Погрешность в x* = {x_star:.4f}')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    max_err = max(errors)
    ax.set_ylim(top=max_err * 5)
    
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2, err * 2.5 if err > 1e-15 else 1e-15,
                f'{err:.2e}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'lab_1_v{variant}.png', dpi=150)
    plt.show()

# endregion

# region Запуск

if __name__ == "__main__":
    v = int(input("Введите номер варианта (1-24): "))
    solve_interpolation(v)

# endregion