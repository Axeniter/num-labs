import numpy as np

functions = {
    1: lambda x: 1.2*x**2 - np.sin(10*x),
    2: lambda x: 2*np.sqrt(x) - np.cos(np.pi*x/2),
    3: lambda x: 2**x - 2*x**2 - 1,
    4: lambda x: 2*np.log(x) - 1/x,
    5: lambda x: 2*np.log10(x) - x/2 + 1,
    6: lambda x: np.log10(x) - 7/(2*x + 6),
    7: lambda x: x*np.log10(x) - 0.5,
    8: lambda x: np.log10(3*x - 1) + np.exp(2*x - 1),
    9: lambda x: np.exp(-x) - 2*(x - 1)**2,
    10: lambda x: 2 - x*np.exp(x),
    11: lambda x: 1/x - np.pi*np.cos(np.pi*x),
    12: lambda x: 1/np.cos(x) - x**2 - 1,
    13: lambda x: 1/np.tan(1.05*x) - x**2,
    14: lambda x: 2*x - np.log10(x) - 7,
    15: lambda x: np.exp(-x) + x**2 - 2,
    16: lambda x: 0.5*x**2 - np.cos(2*x),
    17: lambda x: np.log(0.5*x) - 0.5*np.cos(x),
    18: lambda x: np.log(2*x) - np.exp(2*x),
    19: lambda x: np.exp(-x) + x**3 - 3,
    20: lambda x: 2*x**2 - np.cos(2*x),
    21: lambda x: x**2 - 20*np.sin(x),
    22: lambda x: x**2 - np.sin(5*x),
    23: lambda x: np.log(x) + (x + 1)**3,
    24: lambda x: 2.2*x - 2**x
}

first_derivatives = {
    1: lambda x: 2.4*x - 10*np.cos(10*x),
    2: lambda x: 1/np.sqrt(x) + (np.pi/2)*np.sin(np.pi*x/2),
    3: lambda x: 2**x * np.log(2) - 4*x,
    4: lambda x: 2/x + 1/x**2,
    5: lambda x: 2/(x*np.log(10)) - 0.5,
    6: lambda x: 1/(x*np.log(10)) + 14/(2*x + 6)**2,
    7: lambda x: np.log10(x) + 1/np.log(10),
    8: lambda x: 3/((3*x - 1)*np.log(10)) + 2*np.exp(2*x - 1),
    9: lambda x: -np.exp(-x) - 4*(x - 1),
    10: lambda x: -np.exp(x)*(x + 1),
    11: lambda x: -1/x**2 + np.pi**2 * np.sin(np.pi*x),
    12: lambda x: np.sin(x)/np.cos(x)**2 - 2*x,
    13: lambda x: -1.05/np.sin(1.05*x)**2 - 2*x,
    14: lambda x: 2 - 1/(x*np.log(10)),
    15: lambda x: -np.exp(-x) + 2*x,
    16: lambda x: x + 2*np.sin(2*x),
    17: lambda x: 1/x + 0.5*np.sin(x),
    18: lambda x: 1/x - 2*np.exp(2*x),
    19: lambda x: -np.exp(-x) + 3*x**2,
    20: lambda x: 4*x + 2*np.sin(2*x),
    21: lambda x: 2*x - 20*np.cos(x),
    22: lambda x: 2*x - 5*np.cos(5*x),
    23: lambda x: 1/x + 3*(x + 1)**2,
    24: lambda x: 2.2 - 2**x * np.log(2)
}

second_derivatives = {
    1: lambda x: 2.4 + 100*np.sin(10*x),
    2: lambda x: -0.5*x**(-1.5) + (np.pi**2/4)*np.cos(np.pi*x/2),
    3: lambda x: 2**x * (np.log(2))**2 - 4,
    4: lambda x: -2/x**2 - 2/x**3,
    5: lambda x: -2/(x**2 * np.log(10)),
    6: lambda x: -1/(x**2 * np.log(10)) - 56/(2*x + 6)**3,
    7: lambda x: 1/(x*np.log(10)),
    8: lambda x: -9/((3*x - 1)**2 * np.log(10)) + 4*np.exp(2*x - 1),
    9: lambda x: np.exp(-x) - 4,
    10: lambda x: -np.exp(x)*(x + 2),
    11: lambda x: 2/x**3 + np.pi**3 * np.cos(np.pi*x),
    12: lambda x: (1 + np.sin(x)**2)/np.cos(x)**3 - 2,
    13: lambda x: 2.205*np.cos(1.05*x)/np.sin(1.05*x)**3 - 2,
    14: lambda x: 1/(x**2 * np.log(10)),
    15: lambda x: np.exp(-x) + 2,
    16: lambda x: 1 + 4*np.cos(2*x),
    17: lambda x: -1/x**2 + 0.5*np.cos(x),
    18: lambda x: -1/x**2 - 4*np.exp(2*x),
    19: lambda x: np.exp(-x) + 6*x,
    20: lambda x: 4 + 4*np.cos(2*x),
    21: lambda x: 2 + 20*np.sin(x),
    22: lambda x: 2 + 25*np.cos(5*x),
    23: lambda x: -1/x**2 + 6*(x + 1),
    24: lambda x: -2**x * (np.log(2))**2
}

def get_function_data(variant: int):
    if variant not in functions:
        raise ValueError("Вариант должен быть в пределах 1-24")
    return {
        'function': functions[variant],
        'first_derivative': first_derivatives[variant],
        'second_derivative': second_derivatives[variant]
    }


def get_search_domain(variant: int):
    """Возвращает границы (x_min, x_max) для поиска корней"""
    positive_roots = {1, 3, 5, 11, 12, 13, 14, 19}
    negative_roots = {15}
    all_roots = {2, 4, 6, 7, 8, 9, 10, 16, 17, 18, 20, 21, 22, 23, 24}

    if variant in positive_roots:
        return (0.001, 10.0)
    elif variant in negative_roots:
        return (-10.0, -0.001)
    else:
        return (-10.0, 10.0)