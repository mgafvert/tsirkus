import tsirkus as ts
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn 
from matplotlib.colors import LogNorm



#t = ts.Tsirkus(jumps=[])
# t = ts.Tsirkus(jumps=[(100,10)])
t = ts.Tsirkus()
p0 = np.eye(t.N,1) # init position probability vector
# calculate up to 0.999 probability to win
p_final = 0.999
P = np.squeeze(np.array([p for p in t.evolve_final(p0, p_final)]))
p_end = P[:,-1] # extract end position probability vector
p_end_diff = np.diff(p_end,prepend=0.) # calculate probability p_end_diff[i] to win in i turns
x = np.arange(len(p_end))

# console output
i = np.where(p_end > 0.)[0][0] # first [0] to pick the array from the tuple, second [0] to pick the first element
print(f'Minimum turns to terminate: {i} (probability {p_end[i]:.3e})')

i = np.argmax(p_end_diff)
print(f'Nunber of turns with highest probability to terminate: {i} (probability {p_end_diff[i]:.3e})')

percentiles = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
print(f"{'Percentile':>12}{'Turn (-)':>12}{'Prob (-)':>12}{'Turn (+)':>12}{'Prob (+)':>12}")
for p in percentiles:
    i = np.where(p_end <= p)[0][-1]
    print(f'{p:>12.3f}{i:>12d}{p_end[i]:>12.3e}{i+1:>12d}{p_end[i+1]:>12.3e}')


f1 = plt.figure(1)
plt.plot(p_end)
plt.grid(visible=True)
f1.show()

f2 = plt.figure(2)
plt.plot(p_end_diff)
plt.grid(visible=True)
f2.show()


f3 = plt.figure(3)
text_labels = np.arange(1,t.N+1) # init game path labels 
text_labels = np.flipud(text_labels.reshape(t.shape)) # game path bottom to up
text_labels[::2,:] = np.fliplr(text_labels[::2,:]) # game path switching left to right vs right to left
k = 0
while input(f"Press enter to plot turn {k}, q to quit: ") != 'q':
    P_board = np.copy(P[k,:])
    P_board = np.flipud(P_board.reshape(t.shape))
    P_board[::2,:] = np.fliplr(P_board[::2,:])
    f3.clf()
    plt.title(f"Turn {k} with termination probability {P[k,-1]:.3e}")
    for i in range(text_labels.shape[0]):
        for j in range(text_labels.shape[1]):
            plt.text(j+.5, i+.5, text_labels[i, j], ha="center", va="center", color="g")
    sn.heatmap(data = P_board, cmap = 'hot', xticklabels = False, yticklabels = False)

    f3.show()
    del P_board
    k += 1





