import numpy as np
from table_1 import get_function_data

# region Квадратурные формулы

def left_rectangles(f, a, b, n):
    """Составная формула левых прямоугольников"""
    h = (b - a) / n
    x = np.linspace(a, b - h, n)
    return h * np.sum(f(x))


def right_rectangles(f, a, b, n):
    """Составная формула правых прямоугольников"""
    h = (b - a) / n
    x = np.linspace(a + h, b, n)
    return h * np.sum(f(x))


def midpoint_rectangles(f, a, b, n):
    """Составная формула центральных прямоугольников"""
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
    return h * np.sum(f(x))


def trapezoid(f, a, b, n):
    """Составная формула трапеций"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])


def simpson(f, a, b, n):
    """Формула Симпсона (n должно быть чётным)"""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h/3 * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]))


def weddle(f, a, b, n):
    """Формула Веддля (n должно быть кратно 6)"""
    if n % 6 != 0:
        n = ((n // 6) + 1) * 6
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    coeffs = np.array([1, 5, 1, 6, 1, 5, 1])
    result = 0.0
    for i in range(0, n, 6):
        result += np.sum(coeffs * y[i:i+7])
    
    return 0.3 * h * result


def newton_cotes(f, a, b, n):
    """Формулы Ньютона-Котеса (n = 1..6)"""
    c_tables = {
        1: np.array([1/2, 1/2]),
        2: np.array([1/6, 4/6, 1/6]),
        3: np.array([1/8, 3/8, 3/8, 1/8]),
        4: np.array([7/90, 32/90, 12/90, 32/90, 7/90]),
        5: np.array([19/288, 75/288, 50/288, 50/288, 75/288, 19/288]),
        6: np.array([41/840, 216/840, 27/840, 272/840, 27/840, 216/840, 41/840])
    }
    
    c = c_tables[n]
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (b - a) * np.sum(c * y)


def gauss(f, a, b, n):
    """Формулы Гаусса (n = 1..4)"""
    t_tables = {
        1: np.array([0.0]),
        2: np.array([-0.577350, 0.577350]),
        3: np.array([-0.774597, 0.0, 0.774597]),
        4: np.array([-0.861136, -0.339981, 0.339981, 0.861136])
    }
    c_tables = {
        1: np.array([2.0]),
        2: np.array([1.0, 1.0]),
        3: np.array([5/9, 8/9, 5/9]),
        4: np.array([0.347855, 0.652145, 0.652145, 0.347855])
    }
    
    t = t_tables[n]
    c = c_tables[n]
    x = (b + a) / 2 + (b - a) / 2 * t
    y = f(x)
    return (b - a) / 2 * np.sum(c * y)

# endregion

# region Уточнение точности

def integrate_with_accuracy(f, a, b, method_func, n0, is_composite=True, eps=1e-6, max_iter=20):
    """
    Вычисление интеграла с заданной точностью
    
    Параметры:
    - f: функция
    - a, b: пределы
    - method_func: функция метода (принимает f, a, b, n)
    - n0: начальное число разбиений
    - is_composite: True - удвоение n; False - деление отрезка пополам
    - eps: точность
    - max_iter: максимум итераций
    """
    if is_composite:
        n = n0
        I_old = method_func(f, a, b, n)
        
        for _ in range(max_iter):
            n *= 2
            I_new = method_func(f, a, b, n)
            if abs(I_new - I_old) <= eps:
                return I_new
            I_old = I_new
        return I_new
    else:
        n = n0
        I_old = method_func(f, a, b, n)
        segments = [(a, b)]
        
        for _ in range(max_iter):
            new_segments = []
            for seg_a, seg_b in segments:
                mid = (seg_a + seg_b) / 2
                new_segments.extend([(seg_a, mid), (mid, seg_b)])
            segments = new_segments
            
            I_new = sum(method_func(f, sa, sb, n) for sa, sb in segments)
            if abs(I_new - I_old) <= eps:
                return I_new
            I_old = I_new
        return I_new

# endregion

# region Найти интеграл

def solve_quadrature(f, a, b, method_id, eps=1e-6):
    """Нахождение определенного интеграла для функции f на [a, b] выбранным методом"""
    
    methods = {
        1: ('Левые прямоугольники', left_rectangles, 10),
        2: ('Правые прямоугольники', right_rectangles, 10),
        3: ('Центральные прямоугольники', midpoint_rectangles, 10),
        4: ('Трапеции', trapezoid, 10),
        5: ('Симпсон', simpson, 10),
        6: ('Веддль', weddle, 6),
        7: ('Ньютон-Котес n=1', newton_cotes, 1),
        8: ('Ньютон-Котес n=2', newton_cotes, 2),
        9: ('Ньютон-Котес n=3', newton_cotes, 3),
        10: ('Ньютон-Котес n=4', newton_cotes, 4),
        11: ('Ньютон-Котес n=5', newton_cotes, 5),
        12: ('Ньютон-Котес n=6', newton_cotes, 6),
        13: ('Гаусс n=1', gauss, 1),
        14: ('Гаусс n=2', gauss, 2),
        15: ('Гаусс n=3', gauss, 3),
        16: ('Гаусс n=4', gauss, 4),
    }
    
    name, func, n = methods[method_id]
    
    print(f"\nВариант {variant} | Метод {method_id}: {name}")
    print(f"{'='*50}")
    print(f"Отрезок [{a}, {b}] | Точность eps = {eps}")
    
    if method_id in range(1, 7):
        result = integrate_with_accuracy(f, a, b, func, n, is_composite=True, eps=eps)
    else:
        result = integrate_with_accuracy(f, a, b, func, n, is_composite=False, eps=eps)
    print(f"  Приближённое значение: {result:.12f}")
    
    return result

# endregion

# region Запуск

if __name__ == "__main__":
    variant = int(input("Введите вариант (1-24): "))
    data = get_function_data(variant)
    f = data['function']
    a, b = data['interval']
    for i in range(1,17):
        solve_quadrature(f, a, b, method_id=i, eps=1e-6)

# endregion