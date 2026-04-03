import numpy as np


functions = {
    1: lambda x: x**2 + np.log(x),
    2: lambda x: x**2 - np.log10(x+2),
    3: lambda x: x**2 + np.log(x) - 4,
    4: lambda x: (x-1)**2 - 0.5*np.exp(x),
    5: lambda x: (x-1)**2 - np.exp(-x),
    6: lambda x: x**3 - np.sin(x),
    7: lambda x: 4*x - np.cos(x),
    8: lambda x: x**2 - np.sin(x),
    9: lambda x: x - np.cos(x),
    10: lambda x: x**2 - np.cos(np.pi*x),
    11: lambda x: x**2 - np.sin(np.pi*x),
    12: lambda x: x**2 - np.cos(0.5*np.pi*x),
    13: lambda x: x - 2*np.cos(0.5*np.pi*x),
    14: lambda x: x - np.sin(np.pi*x),
    15: lambda x: 2*x - np.cos(x),
    16: lambda x: x**2 + np.log(x+5),
    17: lambda x: 0.5*x**2 + np.cos(2*x),
    18: lambda x: x**2 - 0.5*np.exp(-x),
    19: lambda x: x**2 + np.log10(x),
    20: lambda x: x - np.log10(x+2),
    21: lambda x: x**2 - np.log10(0.5*x),
    22: lambda x: x**3 - np.cos(2*x),
    23: lambda x: x**2 + np.cos(np.pi*x/2),
    24: lambda x: x/2 - np.cos(x/2)
}


first_derivatives = {
    1: lambda x: 2*x + 1/x,
    2: lambda x: 2*x - 1/((x+2)*np.log(10)),
    3: lambda x: 2*x + 1/x,
    4: lambda x: 2*(x-1) - 0.5*np.exp(x),
    5: lambda x: 2*(x-1) + np.exp(-x),
    6: lambda x: 3*x**2 - np.cos(x),
    7: lambda x: 4 + np.sin(x),
    8: lambda x: 2*x - np.cos(x),
    9: lambda x: 1 + np.sin(x),
    10: lambda x: 2*x + np.pi*np.sin(np.pi*x),
    11: lambda x: 2*x - np.pi*np.cos(np.pi*x),
    12: lambda x: 2*x + 0.5*np.pi*np.sin(0.5*np.pi*x),
    13: lambda x: 1 + np.pi*np.sin(0.5*np.pi*x),
    14: lambda x: 1 - np.pi*np.cos(np.pi*x),
    15: lambda x: 2 + np.sin(x),
    16: lambda x: 2*x + 1/(x+5),
    17: lambda x: x - 2*np.sin(2*x),
    18: lambda x: 2*x + 0.5*np.exp(-x),
    19: lambda x: 2*x + 1/(x*np.log(10)),
    20: lambda x: 1 - 1/((x+2)*np.log(10)),
    21: lambda x: 2*x - 1/(x*np.log(10)),
    22: lambda x: 3*x**2 + 2*np.sin(2*x),
    23: lambda x: 2*x - 0.5*np.pi*np.sin(0.5*np.pi*x),
    24: lambda x: 0.5 + 0.5*np.sin(x/2)
}


second_derivatives = {
    1: lambda x: 2 - 1/x**2,
    2: lambda x: 2 + 1/((x+2)**2 * np.log(10)),
    3: lambda x: 2 - 1/x**2,
    4: lambda x: 2 - 0.5*np.exp(x),
    5: lambda x: 2 - np.exp(-x),
    6: lambda x: 6*x + np.sin(x),
    7: lambda x: np.cos(x),
    8: lambda x: 2 + np.sin(x),
    9: lambda x: np.cos(x),
    10: lambda x: 2 + np.pi**2 * np.cos(np.pi*x),
    11: lambda x: 2 + np.pi**2 * np.sin(np.pi*x),
    12: lambda x: 2 + 0.25*np.pi**2 * np.cos(0.5*np.pi*x),
    13: lambda x: 0.5*np.pi**2 * np.cos(0.5*np.pi*x),
    14: lambda x: np.pi**2 * np.sin(np.pi*x),
    15: lambda x: np.cos(x),
    16: lambda x: 2 - 1/(x+5)**2,
    17: lambda x: 1 - 4*np.cos(2*x),
    18: lambda x: 2 - 0.5*np.exp(-x),
    19: lambda x: 2 - 1/(x**2 * np.log(10)),
    20: lambda x: 1/((x+2)**2 * np.log(10)),
    21: lambda x: 2 + 1/(x**2 * np.log(10)),
    22: lambda x: 6*x + 4*np.cos(2*x),
    23: lambda x: 2 - 0.25*np.pi**2 * np.cos(0.5*np.pi*x),
    24: lambda x: 0.25*np.cos(x/2)
}


third_derivatives = {
    1: lambda x: 2/x**3,
    2: lambda x: -2/((x+2)**3 * np.log(10)),
    3: lambda x: 2/x**3,
    4: lambda x: -0.5*np.exp(x),
    5: lambda x: np.exp(-x),
    6: lambda x: 6 + np.cos(x),
    7: lambda x: -np.sin(x),
    8: lambda x: np.cos(x),
    9: lambda x: -np.sin(x),
    10: lambda x: -np.pi**3 * np.sin(np.pi*x),
    11: lambda x: np.pi**3 * np.cos(np.pi*x),
    12: lambda x: -0.125*np.pi**3 * np.sin(0.5*np.pi*x),
    13: lambda x: -0.25*np.pi**3 * np.sin(0.5*np.pi*x),
    14: lambda x: np.pi**3 * np.cos(np.pi*x),
    15: lambda x: -np.sin(x),
    16: lambda x: 2/(x+5)**3,
    17: lambda x: 8*np.sin(2*x),
    18: lambda x: 0.5*np.exp(-x),
    19: lambda x: 2/(x**3 * np.log(10)),
    20: lambda x: -2/((x+2)**3 * np.log(10)),
    21: lambda x: -2/(x**3 * np.log(10)),
    22: lambda x: 6 - 8*np.sin(2*x),
    23: lambda x: 0.125*np.pi**3 * np.sin(0.5*np.pi*x),
    24: lambda x: -0.125*np.sin(x/2)
}


intervals = {
    1: (0.4, 0.9),
    2: (0.5, 1.0),
    3: (1.5, 2.0),
    4: (0.1, 0.6),
    5: (1.0, 1.5),
    6: (0.6, 1.1),
    7: (0.1, 0.6),
    8: (0.5, 1.0),
    9: (0.5, 1.0),
    10: (0.1, 0.6),
    11: (0.4, 0.9),
    12: (0.4, 0.9),
    13: (0.4, 0.9),
    14: (0.6, 1.1),
    15: (0.1, 0.6),
    16: (0.5, 1.0),
    17: (0.6, 1.1),
    18: (0.1, 0.6),
    19: (0.4, 0.9),
    20: (0.5, 1.0),
    21: (0.5, 1.0),
    22: (0.1, 0.6),
    23: (0.1, 0.6),
    24: (0.4, 0.9)
}


points = {
    1: (0.52, 0.42, 0.87, 0.67),
    2: (0.53, 0.52, 0.97, 0.73),
    3: (1.52, 1.52, 1.97, 1.77),
    4: (0.13, 0.12, 0.57, 0.33),
    5: (1.07, 1.02, 1.47, 1.27),
    6: (0.92, 0.62, 1.07, 0.83),
    7: (0.37, 0.12, 0.57, 0.37),
    8: (0.77, 0.52, 0.97, 0.73),
    9: (0.92, 0.53, 0.98, 0.77),
    10: (0.37, 0.12, 0.58, 0.33),
    11: (0.53, 0.43, 0.86, 0.67),
    12: (0.64, 0.42, 0.87, 0.63),
    13: (0.71, 0.43, 0.87, 0.67),
    14: (0.88, 0.63, 1.08, 0.83),
    15: (0.44, 0.13, 0.58, 0.37),
    16: (0.73, 0.52, 0.97, 0.73),
    17: (0.84, 0.62, 1.07, 0.83),
    18: (0.37, 0.12, 0.58, 0.33),
    19: (0.53, 0.43, 0.86, 0.67),
    20: (0.77, 0.52, 0.97, 0.73),
    21: (0.92, 0.53, 0.98, 0.77),
    22: (0.37, 0.12, 0.58, 0.33),
    23: (0.13, 0.12, 0.57, 0.33),
    24: (0.64, 0.42, 0.87, 0.63)
}


def get_function_data(variant):
    if variant not in functions:
        raise ValueError(f"Вариант должен быть в пределах 1-24")
    
    return {
        'function': functions[variant],
        'first_derivative': first_derivatives[variant],
        'second_derivative': second_derivatives[variant],
        'third_derivative': third_derivatives[variant],
        'interval': intervals[variant],
        'points': points[variant]
    }