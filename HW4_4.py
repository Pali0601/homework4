import numpy as np

# 複合辛普森法則
def composite_simpson(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    fx = f(x)
    result = (h / 3) * (fx[0] + 4 * np.sum(fx[1:n:2]) + 2 * np.sum(fx[2:n-1:2]) + fx[n])
    return result

# 積分 a: \int_0^1 x^{-1/4} \sin x \, dx
# 變量變換 t = x^{-1}，積分變為 \int_1^\infty t^{-7/4} \sin(t^{-1}) \, dt
# 截斷到 [1, 100]
def g1(t):
    return t**(-7/4) * np.sin(t**(-1))

result_a = composite_simpson(g1, 1, 100, 4)
print(f"a. Integral (0 to 1) x^(-1/4) sin x dx: {result_a:.6f}")

# 積分 b: \int_1^\infty x^{-4} \sin x \, dx
# 變量變換 t = x^{-1}，積分變為 \int_0^1 t^2 \sin(t^{-1}) \, dt
# 截斷到 [0.01, 1]
def g2(t):
    return t**2 * np.sin(t**(-1))

result_b = composite_simpson(g2, 0.01, 1, 4)
print(f"b. Integral (1 to inf) x^(-4) sin x dx: {result_b:.6f}")