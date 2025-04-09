import numpy as np
from scipy.special import roots_legendre
from scipy.integrate import quad

# 定義被積函數
def f(x):
    return x**2 * np.log(x)

# 高斯求積法
def gaussian_quadrature(f, a, b, n):
    # 獲取高斯-勒讓德節點和權重
    t, w = roots_legendre(n)
    
    # 變量變換：t -> x
    x = (b - a) / 2 * t + (a + b) / 2
    # 積分公式
    result = (b - a) / 2 * np.sum(w * f(x))
    return result

# 積分區間
a = 1.0
b = 1.5

# n = 3
result_n3 = gaussian_quadrature(f, a, b, 3)
print(f"Gaussian Quadrature (n=3): {result_n3:.6f}")

# n = 4
result_n4 = gaussian_quadrature(f, a, b, 4)
print(f"Gaussian Quadrature (n=4): {result_n4:.6f}")

# 真實值
true_value, _ = quad(f, a, b)
print(f"True value: {true_value:.6f}")

# 計算相對誤差
relative_error_n3 = abs(result_n3 - true_value) / abs(true_value)
relative_error_n4 = abs(result_n4 - true_value) / abs(true_value)

# 轉換為百分比
relative_error_n3_percent = relative_error_n3 * 100
relative_error_n4_percent = relative_error_n4 * 100

print(f"Relative error (n=3, %): {relative_error_n3_percent:.6f}%")
print(f"Relative error (n=4, %): {relative_error_n4_percent:.6f}%")