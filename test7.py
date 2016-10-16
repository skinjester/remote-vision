import numpy as np
from numpy import convolve
import matplotlib.pyplot as plt
 
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
 
x = [1,2,3,4,5,6,7,8,9,10]
y = [3,5,1,8,2,1,6,1,9,2]
 
yMA = movingaverage(y,5)
plt.plot(x[len(x)-len(yMA):], yMA)
plt.plot(x,y)
plt.show()
