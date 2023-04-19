import math
import numpy as np

fs = 8000
fl = 0
fh = fs / 2
bl = 1125 * math.log(1 + fl/700)
bh = 1125 * math.log(1 + fh/700)
p = 24
N = 256
MelF = np.linspace(0, bh-bl, p+2)
F = 700 * (np.exp(MelF/1125) - 1)
df = fs / N
n = int(N/2 + 1)
f = np.array(list(range(n))) * df
bank = np.zeros(24 * n).reshape((24, n))
for i, m in enumerate(list(range(1, p+1))):
    F_left = F[m-1]
    F_mid = F[m]
    F_right = F[m+1]
    print("%3d: %7.2f(Hz), %7.2f(Hz), %7.2f(Hz)" % (i + 1, F_left, F_mid, F_right))
    n_left = math.ceil(F_left / df)
    n_mid = math.ceil(F_mid / df)
    n_right = math.ceil(F_right / df)

