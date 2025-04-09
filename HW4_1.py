import numpy as np

# 定義被積函數
def f(x):
    return np.exp(x) * np.sin(4 * x)

# 給定參數
a = 1.0  # 積分下限
b = 2.0  # 積分上限
h = 0.1  # 步長
n = int((b - a) / h)  # 子區間數量，n = 10

# 積分點
x = np.linspace(a, b, n + 1)  # x = [1.0, 1.1, ..., 2.0]
f_values = f(x)  # 計算 f(x) 在每個點的值

# a. 複合梯形法則
def composite_trapezoidal_rule(f_values, h, n):
    result = (h / 2) * (f_values[0] + 2 * np.sum(f_values[1:n]) + f_values[n])
    return result

# b. 複合辛普森法則
def composite_simpson_rule(f_values, h, n):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    result = (h / 3) * (f_values[0] + 4 * np.sum(f_values[1:n:2]) + 2 * np.sum(f_values[2:n-1:2]) + f_values[n])
    return result

# c. 複合中點法則
def composite_midpoint_rule(f, x, h, n):
    midpoints = (x[:-1] + x[1:]) / 2  # 中點：(x_i + x_{i+1})/2
    result = h * np.sum(f(midpoints))
    return result

# 計算結果
trapezoidal_result = composite_trapezoidal_rule(f_values, h, n)
simpson_result = composite_simpson_rule(f_values, h, n)
midpoint_result = composite_midpoint_rule(f, x, h, n)

# 輸出結果
print(f"a. Composite Trapezoidal Rule: {trapezoidal_result:.6f}")
print(f"b. Composite Simpson's Rule: {simpson_result:.6f}")
print(f"c. Composite Midpoint Rule: {midpoint_result:.6f}")

# 為了驗證，計算真實值（可通過數值積分或解析解）
from scipy.integrate import quad
true_value, _ = quad(f, a, b)
print(f"\nTrue value (for reference): {true_value:.6f}")