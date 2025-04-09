import numpy as np
from scipy.integrate import dblquad

# 定義被積函數
def f(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

# 複合辛普森法則（二重積分）
def composite_simpson_2d(f, a, b, c, d, n, m):
    hx = (b - a) / n
    x = np.linspace(a, b, n + 1)
    I = np.zeros(n + 1)
    for i in range(n + 1):
        xi = x[i]
        ci = c(xi)
        di = d(xi)
        hy = (di - ci) / m
        y = np.linspace(ci, di, m + 1)
        fy = f(xi, y)
        I[i] = (hy / 3) * (fy[0] + 4 * np.sum(fy[1:m:2]) + 2 * np.sum(fy[2:m-1:2]) + fy[m])
    result = (hx / 3) * (I[0] + 4 * np.sum(I[1:n:2]) + 2 * np.sum(I[2:n-1:2]) + I[n])
    return result

# 高斯求積法（二重積分）
def gaussian_quadrature_2d(f, a, b, c, d, n, m):
    from scipy.special import roots_legendre
    tx, wx = roots_legendre(n)
    x = (b - a) / 2 * tx + (a + b) / 2
    I = np.zeros(n)
    for i in range(n):
        xi = x[i]
        ci = c(xi)
        di = d(xi)
        ty, wy = roots_legendre(m)
        y = (di - ci) / 2 * ty + (ci + di) / 2
        fy = f(xi, y)
        I[i] = (di - ci) / 2 * np.sum(wy * fy)
    result = (b - a) / 2 * np.sum(wx * I)
    return result

# 積分區間
a = 0
b = np.pi / 4
c = lambda x: np.sin(x)
d = lambda x: np.cos(x)

# a. 複合辛普森法則 (n=4, m=4)
simpson_result = composite_simpson_2d(f, a, b, c, d, 4, 4)
print(f"a. Composite Simpson's Rule (n=4, m=4): {simpson_result:.6f}")

# b. 高斯求積法 (n=3, m=3)
gauss_result = gaussian_quadrature_2d(f, a, b, c, d, 3, 3)
print(f"b. Gaussian Quadrature (n=3, m=3): {gauss_result:.6f}")

# c. 真實值
true_value, _ = dblquad(lambda y, x: f(x, y), a, b, c, d)
print(f"c. True value: {true_value:.6f}")

# 計算相對誤差並轉換為百分比
simpson_relative_error = abs(simpson_result - true_value) / abs(true_value)
gauss_relative_error = abs(gauss_result - true_value) / abs(true_value)
simpson_relative_error_percent = simpson_relative_error * 100
gauss_relative_error_percent = gauss_relative_error * 100

print(f"\nRelative error (Simpson, %): {simpson_relative_error_percent:.6f}%")
print(f"Relative error (Gaussian, %): {gauss_relative_error_percent:.6f}%")