import tsirkus as ts
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn 
from matplotlib.colors import LogNorm



#t = ts.Tsirkus(jumps=[])
t = ts.Tsirkus()
p0 = np.eye(t.N,1) # init position probability vector
# calculate up to 0.999 probability to win
p_final = 0.999
P = np.squeeze(np.array([p for p in t.evolve_final(p0, p_final)]))
p_end = P[:,-1] # extract end position probability vector
p_end_diff = np.diff(p_end,prepend=0.) # calculate probability p_end_diff[i] to win in i turns
x = np.arange(len(p_end))

f1 = plt.figure(1)
plt.plot(p_end)
plt.grid(visible=True)
f1.show()

f2 = plt.figure(2)
plt.plot(p_end_diff)
plt.grid(visible=True)
f2.show()


f3 = plt.figure(3)

i = 0
while input(f"Press enter to continue turn {i}, q to quit: ") != 'q':
    P_board = np.copy(P[i,:])
    P_board = np.flipud(P_board.reshape(t.shape))
    P_board[::2,:] = np.fliplr(P_board[::2,:])
    f3.clf()
    plt.title(f"Turn {i} with termination probability {P[i,-1]:.3e}")
   # hm = sn.heatmap(data = P_board, cmap = 'hot', xticklabels = False, yticklabels = False, norm = LogNorm())
    hm = sn.heatmap(data = P_board, cmap = 'hot', xticklabels = False, yticklabels = False)
    f3.show()
    # del P_board
    i += 1




# console output
i = np.where(p_end > 0.)[0][0] # first [0] to pick the array from the tuple, second [0] to pick the first element
print(f'Minimum turns to win: {i} (probability {p_end[i]:.3e})')

i = np.argmax(p_end_diff)
print(f'Nunber of turns with highest probability to win: {i} (probability {p_end_diff[i]:.3e})')

percentiles = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
print(f"{'Percentile':>12}{'Turn (-)':>12}{'Prob (-)':>12}{'Turn (+)':>12}{'Prob (+)':>12}")
for p in percentiles:
    i = np.where(p_end <= p)[0][-1]
    print(f'{p:>12.3f}{i:>12d}{p_end[i]:>12.3e}{i+1:>12d}{p_end[i+1]:>12.3e}')

