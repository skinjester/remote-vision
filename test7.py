import numpy as np
from numpy import convolve
import matplotlib.pyplot as plt
 
def movingaverage (values, window):
    weights = np.linspace(10,1, num=window)/window
    print 'weights {}'.format(weights)
    sma = np.convolve(values, weights, 'valid')
    return sma
 
x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]
 
yMA = movingaverage(x,5)
plt.plot(x[len(x)-len(yMA):], yMA)
plt.show()
