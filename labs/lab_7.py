import numpy as np
from typing import Dict, Callable, Tuple, List
from table_5 import get_function_data, get_search_domain
import matplotlib.pyplot as plt


# region Отделение корней

def check_interval(
    f: Callable,
    df: Callable,
    a: float,
    b: float,
    depth: int = 0,
    max_depth: int = 10
) -> List[Tuple[float, float]]:
    """
    Рекурсивно проверяет интервал [a,b] на единственность корня.
    Если производная меняет знак (несколько корней), делит интервал пополам
    и проверяет каждую половину. Продолжает до max_depth или пока не
    получит интервалы с монотонной производной
    """
    x_check = np.linspace(a, b, 30)
    try:
        df_vals = df(x_check)
        df_signs = np.sign(df_vals)
        
        if np.all(df_signs == df_signs[0]) or depth >= max_depth:
            return [(a, b)]
        
        mid = (a + b) / 2
        result = []
        
        if f(a) * f(mid) < 0:
            result.extend(
                check_interval(f, df, a, mid, depth + 1, max_depth)
            )
        elif f(mid) == 0:
            result.append((mid, mid))
        
        if f(mid) * f(b) < 0:
            result.extend(
                check_interval(f, df, mid, b, depth + 1, max_depth)
            )
        
        return result
    except:
        return [(a, b)]


def merge_overlapping_intervals(
    intervals: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Объединяет перекрывающиеся или соприкасающиеся интервалы"""
    if not intervals:
        return []
    
    sorted_ints = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_ints[0]]
    
    for a, b in sorted_ints[1:]:
        last_a, last_b = merged[-1]
        
        if a <= last_b:
            merged[-1] = (last_a, max(last_b, b))
        else:
            merged.append((a, b))
    
    return merged


def separate_all_roots(
    f: Callable,
    df: Callable,
    x_min: float,
    x_max: float,
    scan_step: float = 0.01,
) -> List[Tuple[float, float]]:
    """Выделяет отрезки, содержащие ровно 1 корень"""
    x_scan = np.arange(x_min, x_max + scan_step, scan_step)
    y_scan = np.full_like(x_scan, np.nan)

    for i, x in enumerate(x_scan):
        try:
            y_scan[i] = f(x)
        except (ValueError, ZeroDivisionError):
            pass

    valid = ~np.isnan(y_scan) & ~np.isinf(y_scan)
    x_scan = x_scan[valid]
    y_scan = y_scan[valid]

    if len(x_scan) < 2:
        return []

    signs = np.sign(y_scan)
    intervals = []

    for i in range(len(x_scan) - 1):
        if signs[i] == 0:
            intervals.append((x_scan[i], x_scan[i]))
        elif signs[i] != signs[i + 1]:
            a, b = x_scan[i], x_scan[i + 1]
            sub_intervals = check_interval(f, df, a, b)
            intervals.extend(sub_intervals)

    return merge_overlapping_intervals(intervals)

# endregion

# region Метод хорд и касательных

def choose_ends(f: Callable, ddf: Callable, a: float, b: float) -> Tuple[float, float]:
    """Касательные с того конца, где f(x)*f''(x) > 0"""
    try:
        if f(a) * ddf(a) > 0:
            return a, b
        else:
            return b, a
    except:
        return a, b


def chord_tangent_method(
    f: Callable,
    df: Callable,
    ddf: Callable,
    a: float,
    b: float,
    eps: float = 1e-6,
    max_iter: int = 100
) -> float:
    """Уточнение корня комбинированным методом хорд и касательных"""
    x_t, x_c = choose_ends(f, ddf, a, b)

    for k in range(1, max_iter + 1):
        f_t = f(x_t)
        f_c = f(x_c)
        df_t = df(x_t)

        if abs(df_t) < 1e-15:
            break

        x_t_new = x_t - f_t / df_t

        denom = f_t - f_c
        if abs(denom) < 1e-15:
            break
        x_c_new = x_c - f_c * (x_t - x_c) / denom

        if abs(x_t_new - x_c_new) < eps:
            root = (x_t_new + x_c_new) / 2
            return root

        x_t, x_c = x_t_new, x_c_new

    root = (x_t + x_c) / 2
    return root

# endregion

# region Решение варианта

def solve_variant(variant: int, eps: float = 1e-6) -> Dict:
    """Находит все корни для выбранного варианта"""
    data = get_function_data(variant)
    f, df, ddf = data['function'], data['first_derivative'], data['second_derivative']
    x_min, x_max = get_search_domain(variant)

    print(f"ВАРИАНТ {variant}")
    print(f"{'='*50}")
    print(f"Область поиска: [{x_min}, {x_max}]")

    print(f"\nШаг 1: Отделение корней")
    print(f"{'-'*50}")
    intervals = separate_all_roots(f, df, x_min, x_max)

    if not intervals:
        print("Корни не найдены")
        return {'variant': variant, 'function': f, 'x_min': x_min, 'x_max': x_max, 'roots': []}

    print(f"Найдено интервалов: {len(intervals)}")
    for i, (a, b) in enumerate(intervals, 1):
        print(f"  [{a:.6f}, {b:.6f}]")

    print(f"\nШаг 2: Уточнение корней")
    print(f"{'-'*50}")
    roots = []
    for i, (a, b) in enumerate(intervals, 1):
        if a == b:
            root = a
            print(f"  Корень {i}: x = {root:.10f} | f(x) = {abs(f(root)):.2e}")
        else:
            try:
                root = chord_tangent_method(f, df, ddf, a, b, eps)
                print(f"  Корень {i}: x = {root:.10f} | f(x) = {abs(f(root)):.2e}")
            except Exception as e:
                print(f"  Ошибка на [{a}, {b}]: {e}")
                continue
        roots.append(root)

    if variant == 24 and roots:
        roots = [min(roots)]
        print(f"  Наименьший корень: {roots[0]:.10f}")

    print(f"\nИтог")
    print(f"{'-'*50}")
    print(f"Найдено корней: {len(roots)}")
    for i, r in enumerate(roots, 1):
        print(f"  x{i} = {r:.10f}")

    return {
        'variant': variant,
        'function': f,
        'x_min': x_min,
        'x_max': x_max,
        'roots': roots
    }


def plot_variant(result: Dict, filename: str = "lab_7.png"):
    """Строит график по результату solve_variant и сохраняет в файл"""
    f = result['function']
    x_min, x_max = result['x_min'], result['x_max']
    roots = result['roots']
    variant = result['variant']

    x_plot = np.linspace(x_min, x_max, 2000)
    y_plot = np.full_like(x_plot, np.nan)

    for i, x in enumerate(x_plot):
        try:
            y = f(x)
            if np.isfinite(y):
                y_plot[i] = y
        except:
            pass

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_plot, y_plot, 'b-', linewidth=1.5, label=f'f(x) — Вариант {variant}')

    ax.axhline(y=0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)

    if roots:
        ax.scatter(roots, [0]*len(roots), color='red', s=100, zorder=5,
                   label=f'Корни ({len(roots)} шт.)')
        
        for i, root in enumerate(roots):
            ax.annotate(
                f'{root:.4f}',
                xy=(root, 0),
                xytext=(0, -20 - 15*(i % 3)),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Вариант {variant} | Корней: {len(roots)}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

# endregion

# region Запуск

if __name__ == "__main__":
    v = int(input("Введите вариант: "))
    result = solve_variant(v, eps=1e-6)
    plot_variant(result, filename="lab_7.png")

# endregion
