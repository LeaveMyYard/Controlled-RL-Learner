import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as P

# import scipy.interpolate as spitp
import scipy.optimize as spopt


def Vandermond(X, F):
    A = np.vander(X, increasing=True)
    return np.linalg.pinv(A).dot(F)


def ChebishevKnots(n, a, b):
    x = np.zeros((n + 1,))
    # define x[i]
    return x


def Rn(x0, x, d4f_max):
    """
    Вычисление оценки погрешности интерполирования
    для инт. мн-на 3-й степени в точке x0
    x -- массив узлов интерполяции
    d4f_max -- максимум модуля 4-й произв-й интерполируемой функции
    """
    y = d4f_max
    for i in range(len(x)):
        y *= (x0 - x[i]) / (i + 1)
    return abs(y)


a = -np.pi
b = np.pi
x = sym.Symbol("x")
f = lambda x: x * np.cos(x)

XX = np.linspace(a, b, num=50)
n = 10  # change n=3,5,10
X = np.linspace(a, b, num=n + 1)
Xch = ChebishevKnots(n + 1, a, b)
F = f(X)
Fch = f(Xch)
c = Vandermond(X, F)
Lx = P.polyval(XX, c)
cch = Vandermond(Xch, Fch)
Lxch = P.polyval(XX, cch)
# l=spitp.lagrange(X,F)
# lx=l(XX)
fig, ax = plt.subplots()
ax.plot(XX, f(XX), color="blue", ls="--", label="original")
ax.plot(XX, Lx, color="red", label="Lagrange")
# ax.plot(XX,lx,color='yellow',ls=':',label='scipy')
ax.scatter(X, F, color="red", marker="o", label="knots")
ax.plot(XX, Lxch, color="green", label="Lagrange with Chebyshev")
ax.scatter(Xch, Fch, color="green", marker="o", label="Chebyshev knots")
ax.grid(True)
ax.legend(loc=0)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Интерполяционный многочлен степени {}".format(n))

# оценка (n+1)-ой производной искомой функции по модулю на промежутке [a,b]
dnf = sym.diff(x * sym.cos(x), x, n + 1)
df = lambda x: float(dnf.subs({"x": x}))
x_min = spopt.brent(df, brack=(a, b))
x_min1 = spopt.brent(lambda x: -df(x), brack=(a, b))
dnf_max = max(abs(df(x_min)), abs(df(x_min1)))

h = (b - a) / n
x0 = np.arange(a + h / 2, b, h)  # ~x(i)=(x(i-1)+x(i))/2
r_fuct = abs(f(x0) - P.polyval(x0, c))
r_fuctCh = abs(f(x0) - P.polyval(x0, cch))
r_apr = np.zeros((n,))
r_aprCh = np.zeros((n,))
for i in range(n):
    r_apr[i] = Rn(
        x0[i], X, dnf_max
    )  # оценки погрешности интерполирования в точках ~x(i)
    r_aprCh[i] = Rn(x0[i], Xch, dnf_max)
fig, ax = plt.subplots()
ax.plot(x0, r_fuct, "bx", label="факт.")
ax.plot(x0, r_apr, "rx", label="априорная")
ax.plot(x0, r_fuctCh, "cx", label="факт.Чебышева")
ax.plot(x0, r_aprCh, "gx", label="априорная Чебышева")
ax.grid(True)
ax.legend(loc=0)
ax.set_xlabel("x")
ax.set_title("Погрешности  интерполяции мн-м {}-й степени".format(n))
plt.show()
