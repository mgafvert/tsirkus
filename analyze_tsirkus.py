import tsirkus as ts
import numpy as np
import matplotlib.pyplot as plt


#t = ts.Tsirkus(jumps=[])
t = ts.Tsirkus()
p0 = np.eye(t.N,1)
p_final = 0.999
P = np.squeeze(np.array([p for p in t.evolve_final(p0, p_final)]))
p_end = P[:,-1]
p_end_diff = np.diff(p_end,prepend=0.)
x = np.arange(len(p_end))
f1 = plt.figure(1)
plt.plot(p_end)
plt.grid(visible=True)
f1.show()

f2 = plt.figure(2)
plt.plot(p_end_diff)
plt.grid(visible=True)
f2.show()

i = np.where(p_end > 0.)[0][0] # first [0] to pick the array from the tuple, second [0] to pick the first element
print(f'Minimum turns to win: {i} (probability {p_end[i]:.3e})')

i = np.argmax(p_end_diff)
print(f'Nunber of turns with highest probability to win: {i} (probability {p_end_diff[i]:.3e})')

percentiles = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
print(f"{'Percentile':>12}{'Turn (-)':>12}{'Prob (-)':>12}{'Turn (+)':>12}{'Prob (+)':>12}")
for p in percentiles:
    i = np.where(p_end <= p)[0][-1]
    print(f'{p:>12.3f}{i:>12d}{p_end[i]:>12.3e}{i+1:>12d}{p_end[i+1]:>12.3e}')

