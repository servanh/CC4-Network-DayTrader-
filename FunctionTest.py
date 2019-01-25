import numpy as np
import matplotlib.pyplot as plt
import csv
from CC4 import CC4
from Converter import floatToUnary, unaryToFloat
from sklearn.metrics import mean_squared_error


#Create data
a = np.arange(83)
m = np.sin(0.2*a) + ((0.01*a)*(1-(0.01*a))) + 0.2
size = len(m)
n = np.arange(size)+1
sampleSize = int(size-30)
(float_info, data_un) = floatToUnary(m, 30)

#Create predicted data
stockPredNetwork = CC4(data_un[0:sampleSize], data_un[1:(sampleSize+1)], 2)
predData_un = [stockPredNetwork.feedForward(data_un[i]) for i in range(0,size)]
predData = unaryToFloat(predData_un, float_info)


#Calculate error
MSE = mean_squared_error(m[0:83], predData[0:83])

print("Mean square error for predicted values is %f" % MSE)

#Plot
plt.ylabel("sin(0.2*a) + ((0.01*a)*(1-(0.01*a))) + 0.2")
plt.xlabel("x")
plt.plot(n, m)
plt.plot(n, predData)
plt.legend(["x", "Predicted"])
plt.show()