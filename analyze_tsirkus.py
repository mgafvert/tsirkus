import tsirkus as ts
import numpy as np
import matplotlib.pyplot as plt


t = ts.Tsirkus()
p0 = np.eye(t.N,1)
p_final = 0.99
P = np.squeeze(np.array([p for p in t.evolve_final(p0, p_final)]))
p_end = P[:,-1]
f1 = plt.figure()
plt.plot(p_end)
f1.show()
p_end_diff = np.diff(p_end,prepend=0.)
f2 = plt.figure()
plt.plot(p_end_diff)
f2.show()

