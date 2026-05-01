import numpy as np
from lab_5 import tridiagonal_method
import matplotlib.pyplot as plt
from table_1 import get_function_data

# region Эрмитов сплайн

class HermiteSpline:
    """Эрмитов кубический сплайн"""
    
    def __init__(self, x_nodes: np.ndarray, y_nodes: np.ndarray, dy_nodes: np.ndarray):
        self.x = np.asarray(x_nodes, dtype=float)
        self.y = np.asarray(y_nodes, dtype=float)
        self.dy = np.asarray(dy_nodes, dtype=float)
        
        N = len(self.x) - 1
        self.a = np.zeros(N)
        self.b = np.zeros(N)
        
        for i in range(N):
            h = self.x[i + 1] - self.x[i]
            diff = (self.y[i + 1] - self.y[i]) / h
            
            self.a[i] = (6 / h) * (diff - (2 * self.dy[i] + self.dy[i + 1]) / 3)
            self.b[i] = (12 / h**2) * ((self.dy[i] + self.dy[i + 1]) / 2 - diff)
    
    def value(self, x: float) -> float:
        """Значение сплайна в точке x"""
        for i in range(len(self.x)):
            if x == self.x[i]:
                return self.y[i]
        
        i = np.searchsorted(self.x, x) - 1
        i = max(0, min(i, len(self.x) - 2))
        
        dx = x - self.x[i]
        return self.y[i] + self.dy[i] * dx + self.a[i] * dx**2 / 2 + self.b[i] * dx**3 / 6
    
    def derivative(self, x: float) -> float:
        """Первая производная сплайна в точке x"""
        for i in range(len(self.x)):
            if x == self.x[i]:
                return self.dy[i]
        
        i = np.searchsorted(self.x, x) - 1
        i = max(0, min(i, len(self.x) - 2))
        
        dx = x - self.x[i]
        return self.dy[i] + self.a[i] * dx + self.b[i] * dx**2 / 2
    
# endregion

# region Сплайн через наклоны

class CubicSplineSlopes:
    """Кубический нелокальный сплайн через наклоны"""
    
    def __init__(self, x_nodes: np.ndarray, y_nodes: np.ndarray,
                 boundary_condition: int = 1, bc_values: tuple = None):
        self.x = np.asarray(x_nodes, dtype=float)
        self.y = np.asarray(y_nodes, dtype=float)
        self.N = len(self.x) - 1
        self.h = np.diff(self.x)
        
        self.m = self._compute_slopes(boundary_condition, bc_values)
        
        self.a = np.zeros(self.N)
        self.b = np.zeros(self.N)
        for i in range(self.N):
            diff = (self.y[i+1] - self.y[i]) / self.h[i]
            self.a[i] = (6 / self.h[i]) * (diff - (2*self.m[i] + self.m[i+1]) / 3)
            self.b[i] = (12 / self.h[i]**2) * ((self.m[i] + self.m[i+1]) / 2 - diff)
    
    def _compute_slopes(self, boundary_condition, bc_values):
        n = self.N + 1
        
        lam = np.zeros(n)
        mu = np.zeros(n)
        for i in range(1, n - 1):
            lam[i] = self.h[i-1] / (self.h[i-1] + self.h[i])
            mu[i] = 1 - lam[i]
        
        g = np.zeros(n)
        for i in range(1, n - 1):
            g[i] = 3 * (lam[i] * (self.y[i+1] - self.y[i]) / self.h[i] +
                        mu[i] * (self.y[i] - self.y[i-1]) / self.h[i-1])
        
        if boundary_condition == 1:
            return self._solve_type_1(bc_values[0], bc_values[1], lam, mu, g)
        elif boundary_condition == 2:
            return self._solve_type_2(bc_values[0], bc_values[1], lam, mu, g)
        elif boundary_condition == 3:
            return self._solve_type_3(lam, mu, g)
        elif boundary_condition == 4:
            return self._solve_type_4(lam, mu, g)
        else:
            raise ValueError("Неизвестный тип краевого условия")
    
    def _solve_type_1(self, m0, mN, lam, mu, g):
        """S'(a) = A, S'(b) = B"""
        n = self.N + 1
        L = np.zeros(n - 2)
        D = np.zeros(n - 2)
        U = np.zeros(n - 2)
        F = np.zeros(n - 2)
        
        for i in range(1, n - 1):
            idx = i - 1
            if i == 1:
                D[idx] = 2
                U[idx] = 1
                F[idx] = 3 * (self.y[i] - self.y[i-1]) / self.h[i-1] - m0
            elif i == n - 2:
                L[idx] = 1
                D[idx] = 2
                F[idx] = 3 * (self.y[i+1] - self.y[i]) / self.h[i] - mN
            else:
                L[idx] = mu[i]
                D[idx] = 2
                U[idx] = lam[i]
                F[idx] = g[i]
        
        m_inner = tridiagonal_method(L, D, U, F)
        m = np.zeros(n)
        m[0] = m0
        m[1:-1] = m_inner
        m[-1] = mN
        return m
    
    def _solve_type_2(self, y0pp, yNpp, lam, mu, g):
        """S''(a) = A, S''(b) = B"""
        n = self.N + 1
        L = np.zeros(n)
        D = np.zeros(n)
        U = np.zeros(n)
        F = np.zeros(n)
        
        D[0] = 2
        U[0] = 1
        F[0] = 3 * (self.y[1] - self.y[0]) / self.h[0] - self.h[0] * y0pp / 2
        
        for i in range(1, n - 1):
            L[i] = mu[i]
            D[i] = 2
            U[i] = lam[i]
            F[i] = g[i]
        
        L[-1] = 1
        D[-1] = 2
        F[-1] = 3 * (self.y[-1] - self.y[-2]) / self.h[-1] + self.h[-1] * yNpp / 2
        
        return tridiagonal_method(L, D, U, F)
    
    def _solve_type_3(self, lam, mu, g):
        """Периодический сплайн"""
        n = self.N + 1
        L = np.zeros(n - 1)
        D = np.zeros(n - 1)
        U = np.zeros(n - 1)
        F = np.zeros(n - 1)
        
        for i in range(1, n):
            idx = i - 1
            L[idx] = mu[i]
            D[idx] = 2
            U[idx] = lam[i]
            F[idx] = g[i]
        
        m_inner = tridiagonal_method(L, D, U, F)
        m = np.zeros(n)
        m[1:] = m_inner
        m[0] = m[-1]
        return m
    
    def _solve_type_4(self, lam, mu, g):
        """Непрерывность S''' на соседних с краями отрезках"""
        n = self.N + 1
        L = np.zeros(n - 2)
        D = np.zeros(n - 2)
        U = np.zeros(n - 2)
        F = np.zeros(n - 2)
        
        gamma1 = self.h[0] / self.h[1]
        D[0] = 1 + gamma1
        U[0] = gamma1
        F[0] = (lam[1] * (3 + 2*gamma1) * (self.y[2] - self.y[1]) / self.h[1] +
                mu[1] * (self.y[1] - self.y[0]) / self.h[0])
        
        for i in range(2, n - 2):
            idx = i - 1
            L[idx] = mu[i]
            D[idx] = 2
            U[idx] = lam[i]
            F[idx] = g[i]
        
        gammaN = self.h[-1] / self.h[-2]
        L[-1] = gammaN
        D[-1] = 1 + gammaN
        F[-1] = (mu[n-2] * (3 + 2*gammaN) * (self.y[-2] - self.y[-3]) / self.h[-2] +
                 lam[n-2] * (self.y[-1] - self.y[-2]) / self.h[-1])
        
        m_inner = tridiagonal_method(L, D, U, F)
        m = np.zeros(n)
        m[1:-1] = m_inner
        
        m[0] = (-(1 - gamma1**2) * m[1] + gamma1**2 * m[2] +
                2 * gamma1 * (self.y[1] - self.y[0]) / self.h[0] -
                2 * gamma1**2 * (self.y[2] - self.y[1]) / self.h[1])
        
        m[-1] = (gammaN**2 * m[-3] - (1 - gammaN**2) * m[-2] -
         2 * gammaN**2 * (self.y[-2] - self.y[-3]) / self.h[-2] +
         2 * (self.y[-1] - self.y[-2]) / self.h[-1])
        
        return m
    
    def value(self, x: float) -> float:
        if x == self.x[-1]:
            return self.y[-1]
        
        i = np.searchsorted(self.x, x) - 1
        i = max(0, min(i, self.N - 1))
        
        dx = x - self.x[i]
        return (self.y[i] + self.m[i] * dx + 
                self.a[i] * dx**2 / 2 + self.b[i] * dx**3 / 6)
    
    def derivative(self, x: float) -> float:
        if x == self.x[-1]:
            return self.m[-1]
        
        i = np.searchsorted(self.x, x) - 1
        i = max(0, min(i, self.N - 1))
        
        dx = x - self.x[i]
        return self.m[i] + self.a[i] * dx + self.b[i] * dx**2 / 2
    
    def second_derivative(self, x: float) -> float:
        if x == self.x[-1]:
            i = self.N - 1
            return self.a[i] + self.b[i] * self.h[i]
        
        i = np.searchsorted(self.x, x) - 1
        i = max(0, min(i, self.N - 1))
        
        dx = x - self.x[i]
        return self.a[i] + self.b[i] * dx
    
    def integral(self, a: float, b: float) -> float:
        result = 0.0
        
        for i in range(self.N):
            left = max(a, self.x[i])
            right = min(b, self.x[i+1])
            
            if left >= right:
                continue
            
            def antideriv(t):
                dt = t - self.x[i]
                return (self.y[i] * dt + self.m[i] * dt**2 / 2 +
                        self.a[i] * dt**3 / 6 + self.b[i] * dt**4 / 24)
            
            result += antideriv(right) - antideriv(left)
        
        return result

# endregion

# region Сплайн через моменты

class CubicSplineMoments:
    """Кубический нелокальный сплайн через моменты"""
    
    def __init__(self, x_nodes: np.ndarray, y_nodes: np.ndarray,
                 boundary_condition: int = 1, bc_values: tuple = None):
        self.x = np.asarray(x_nodes, dtype=float)
        self.y = np.asarray(y_nodes, dtype=float)
        self.N = len(self.x) - 1
        self.h = np.diff(self.x)
        
        self.M = self._compute_moments(boundary_condition, bc_values)
        
        self.c = np.zeros(self.N)
        for i in range(self.N):
            self.c[i] = ((self.y[i+1] - self.y[i]) / self.h[i] 
                         - self.h[i] / 6 * (2 * self.M[i] + self.M[i+1]))
    
    def _compute_moments(self, boundary_condition: int, bc_values: tuple) -> np.ndarray:
        n = self.N + 1
        
        lam = np.zeros(n)
        mu = np.zeros(n)
        for i in range(1, n - 1):
            lam[i] = self.h[i-1] / (self.h[i-1] + self.h[i])
            mu[i] = 1 - lam[i]
        
        g = np.zeros(n)
        for i in range(1, n - 1):
            g[i] = (6 / (self.h[i-1] + self.h[i]) * 
                    ((self.y[i+1] - self.y[i]) / self.h[i] - 
                     (self.y[i] - self.y[i-1]) / self.h[i-1]))
        
        if boundary_condition == 1:
            A, B = bc_values
            return self._solve_type_1(A, B, lam, mu, g)
        elif boundary_condition == 2:
            A, B = bc_values
            return self._solve_type_2(A, B, lam, mu, g)
        elif boundary_condition == 3:
            return self._solve_type_3(lam, mu, g)
        elif boundary_condition == 4:
            return self._solve_type_4(lam, mu, g)
        else:
            raise ValueError("Неизвестный тип краевого условия")
    
    def _solve_type_1(self, A: float, B: float,
                      lam: np.ndarray, mu: np.ndarray,
                      g: np.ndarray) -> np.ndarray:
        """S'(a) = A, S'(b) = B"""
        n = self.N + 1
        
        L = np.zeros(n)
        D = np.zeros(n)
        U = np.zeros(n)
        F = np.zeros(n)
        
        D[0] = 2
        U[0] = 1
        F[0] = 6 / self.h[0] * ((self.y[1] - self.y[0]) / self.h[0] - A)
        
        for i in range(1, n - 1):
            L[i] = lam[i]
            D[i] = 2
            U[i] = mu[i]
            F[i] = g[i]
        
        L[-1] = 1
        D[-1] = 2
        F[-1] = 6 / self.h[-1] * (B - (self.y[-1] - self.y[-2]) / self.h[-1])
        
        return tridiagonal_method(L, D, U, F)
    
    def _solve_type_2(self, A: float, B: float,
                      lam: np.ndarray, mu: np.ndarray,
                      g: np.ndarray) -> np.ndarray:
        """S''(a) = A, S''(b) = B"""
        n = self.N + 1
        
        L = np.zeros(n - 2)
        D = np.zeros(n - 2)
        U = np.zeros(n - 2)
        F = np.zeros(n - 2)
        
        for i in range(1, n - 1):
            idx = i - 1
            
            if i == 1:
                D[idx] = 2
                U[idx] = mu[i]
                F[idx] = g[i] - lam[i] * A
            elif i == n - 2:
                L[idx] = lam[i]
                D[idx] = 2
                F[idx] = g[i] - mu[i] * B
            else:
                L[idx] = lam[i]
                D[idx] = 2
                U[idx] = mu[i]
                F[idx] = g[i]
        
        M_inner = tridiagonal_method(L, D, U, F)
        
        M = np.zeros(n)
        M[0] = A
        M[1:-1] = M_inner
        M[-1] = B
        
        return M
    
    def _solve_type_3(self, lam: np.ndarray, mu: np.ndarray,
                      g: np.ndarray) -> np.ndarray:
        """Периодический сплайн"""
        n = self.N + 1
        
        L = np.zeros(n - 1)
        D = np.zeros(n - 1)
        U = np.zeros(n - 1)
        F = np.zeros(n - 1)
        
        for i in range(1, n):
            idx = i - 1
            L[idx] = lam[i]
            D[idx] = 2
            U[idx] = mu[i]
            F[idx] = g[i]
        
        M_inner = tridiagonal_method(L, D, U, F)
        
        M = np.zeros(n)
        M[1:] = M_inner
        M[0] = M[-1]
        
        return M
    
    def _solve_type_4(self, lam: np.ndarray, mu: np.ndarray,
                      g: np.ndarray) -> np.ndarray:
        """Непрерывность S''' на соседних с краями отрезках"""
        n = self.N + 1
        
        L = np.zeros(n - 2)
        D = np.zeros(n - 2)
        U = np.zeros(n - 2)
        F = np.zeros(n - 2)
        
        gamma1 = self.h[0] / self.h[1]
        D[0] = 2 + gamma1
        U[0] = (self.h[1] - self.h[0]) / self.h[1]
        F[0] = (6 / (self.h[0] + self.h[1]) * 
                ((self.y[2] - self.y[1]) / self.h[1] - 
                 (self.y[1] - self.y[0]) / self.h[0]))
        
        for i in range(2, n - 2):
            idx = i - 1
            L[idx] = lam[i]
            D[idx] = 2
            U[idx] = mu[i]
            F[idx] = g[i]
        
        gammaN = self.h[-1] / self.h[-2]
        L[-1] = (self.h[-2] - self.h[-1]) / self.h[-2]
        D[-1] = 2 + gammaN
        F[-1] = (6 / (self.h[-2] + self.h[-1]) * 
                ((self.y[-1] - self.y[-2]) / self.h[-1] - 
                 (self.y[-2] - self.y[-3]) / self.h[-2]))
        
        M_inner = tridiagonal_method(L, D, U, F)
        
        M = np.zeros(n)
        M[1:-1] = M_inner
        
        M[0] = (1 + gamma1) * M[1] - gamma1 * M[2]
        
        M[-1] = (1 + gammaN) * M[-2] - gammaN * M[-3]
        
        return M
    
    def value(self, x: float) -> float:
        """Значение сплайна в точке x"""
        if x == self.x[-1]:
            return self.y[-1]
        
        i = np.searchsorted(self.x, x) - 1
        i = max(0, min(i, self.N - 1))
        
        dx = x - self.x[i]
        return (self.y[i] + self.c[i] * dx + 
                self.M[i] * dx**2 / 2 + 
                (self.M[i+1] - self.M[i]) / self.h[i] * dx**3 / 6)
    
    def derivative(self, x: float) -> float:
        """Первая производная сплайна в точке x"""
        if x == self.x[-1]:
            i = self.N - 1
            dx = self.h[i]
            return self.c[i] + self.M[i] * dx + (self.M[i+1] - self.M[i]) / self.h[i] * dx**2 / 2
        
        i = np.searchsorted(self.x, x) - 1
        i = max(0, min(i, self.N - 1))
        
        dx = x - self.x[i]
        return (self.c[i] + self.M[i] * dx + 
                (self.M[i+1] - self.M[i]) / self.h[i] * dx**2 / 2)
    
    def second_derivative(self, x: float) -> float:
        if x == self.x[-1]:
            i = self.N - 1
            return self.M[i] + (self.M[i+1] - self.M[i]) / self.h[i] * self.h[i]
        
        i = np.searchsorted(self.x, x) - 1
        i = max(0, min(i, self.N - 1))
        
        dx = x - self.x[i]
        return self.M[i] + (self.M[i+1] - self.M[i]) / self.h[i] * dx
    
    def integral(self, a: float, b: float) -> float:
        result = 0.0
        
        for i in range(self.N):
            left = max(a, self.x[i])
            right = min(b, self.x[i+1])
            
            if left >= right:
                continue
            
            def antideriv_simple(t):
                dt = t - self.x[i]
                return (self.y[i] * dt + self.c[i] * dt**2 / 2 +
                        self.M[i] * dt**3 / 6 +
                        (self.M[i+1] - self.M[i]) / self.h[i] * dt**4 / 24)
            
            result += antideriv_simple(right) - antideriv_simple(left)
        
        return result
    
# endregion

# region Решение варианта

def plot_variant(variant, spline_type='slopes', bc_type=1):
    """
    Построение графиков сравнения сплайна с точной функцией
    
    Параметры:
    - variant: номер варианта (1-24)
    - spline_type: 'slopes' (через наклоны) или 'moments' (через моменты)
    - bc_type: тип краевого условия (1-4)
    """
    
    data = get_function_data(variant)
    f = data['function']
    df = data['first_derivative']
    ddf = data['second_derivative']
    a, b = data['interval']
    
    n_nodes = 10
    x_nodes = np.linspace(a, b, n_nodes)
    y_nodes = f(x_nodes)
    
    if bc_type == 1:
        bc_values = (df(a), df(b))
    elif bc_type == 2:
        bc_values = (ddf(a), ddf(b))
    else:
        bc_values = None
    
    if spline_type == 'slopes':
        spline = CubicSplineSlopes(x_nodes, y_nodes, bc_type, bc_values)
        title_type = 'через наклоны'
    else:
        spline = CubicSplineMoments(x_nodes, y_nodes, bc_type, bc_values)
        title_type = 'через моменты'
    
    x_plot = np.linspace(a, b, 300)
    
    f_exact = f(x_plot)
    f_spline = np.array([spline.value(x) for x in x_plot])
    
    df_exact = df(x_plot)
    df_spline = np.array([spline.derivative(x) for x in x_plot])
    
    ddf_exact = ddf(x_plot)
    ddf_spline = np.array([spline.second_derivative(x) for x in x_plot])
    
    exact_int = np.trapezoid(f_exact, x_plot)
    spline_int = spline.integral(a, b)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    bc_names = {1: "S'(a), S'(b)", 2: "S''(a), S''(b)",
                3: 'периодический', 4: "S''' непрерывна"}
    
    fig.suptitle(f'Вариант {variant} | Сплайн {title_type} | '
                 f'Краевое условие: тип {bc_type} ({bc_names[bc_type]})',
                 fontsize=13, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(x_plot, f_exact, 'b-', linewidth=2, label='f(x) точная')
    ax.plot(x_plot, f_spline, 'r--', linewidth=1.5, label='S(x) сплайн')
    ax.plot(x_nodes, y_nodes, 'ko', markersize=6, label='узлы')
    ax.set_title('Функция f(x)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(x_plot, df_exact, 'b-', linewidth=2, label="f'(x) точная")
    ax.plot(x_plot, df_spline, 'r--', linewidth=1.5, label="S'(x) сплайн")
    ax.plot(x_nodes, df(x_nodes), 'ko', markersize=6, label='узлы')
    ax.set_title("Первая производная f'(x)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(x_plot, ddf_exact, 'b-', linewidth=2, label="f''(x) точная")
    ax.plot(x_plot, ddf_spline, 'r--', linewidth=1.5, label="S''(x) сплайн")
    ax.set_title("Вторая производная f''(x)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    error = np.abs(f_spline - f_exact)
    ax.semilogy(x_plot, error, 'r-', linewidth=1.5)
    ax.set_title('Погрешность |S(x) - f(x)|')
    ax.set_ylabel('абсолютная погрешность')
    ax.grid(True, alpha=0.3)
    
    integral_text = (
        f'∫f(x)dx = {exact_int:.10f}    '
        f'∫S(x)dx = {spline_int:.10f}    '
        f'|Δ| = {abs(exact_int - spline_int):.2e}'
    )
    
    fig.text(0.5, 0.01, integral_text, ha='center', fontsize=16,
             fontfamily='monospace')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(f'lab_6_v{variant}_{spline_type}_bc{bc_type}.png', dpi=150)
    plt.show()

# endregion

# region Запуск

if __name__ == "__main__":
    v = int(input("Введите вариант (1-24): "))
    st = input("Тип сплайна (slopes или moments): ")
    bc = int(input("Тип краевого условия (1-4): "))
    plot_variant(variant=v, spline_type=st, bc_type=bc)

# endregion