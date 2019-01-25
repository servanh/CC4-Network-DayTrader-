import numpy as np
import matplotlib.pyplot as plt
import csv
from CC4 import CC4
from Converter import floatToUnary, unaryToFloat
from sklearn.metrics import mean_squared_error


#Create data
data = []
with open('tsla.csv', 'rt') as csvfile:
    stockreader = csv.reader(csvfile, delimiter=',')
    for row in stockreader:
        data.append(float(row[0]))
size = len(data)
n = np.arange(size)+1
sampleSize = int(size-30)
(float_info, data_un) = floatToUnary(data, 30)

#Create predicted data
stockPredNetwork = CC4(data_un[0:sampleSize], data_un[1:(sampleSize+1)], 2)
predData_un = [stockPredNetwork.feedForward(data_un[i]) for i in range(0,size)]
predData = unaryToFloat(predData_un, float_info)


#Calculate error
MSE = mean_squared_error(data[0:53], predData[0:53])

print("Mean square error for predicted values is %f" % MSE)

#Plot
plt.ylabel("TSLA stock value")
plt.xlabel("Weeks")
plt.plot(n, data)
plt.plot(n, predData)
plt.legend(["Data", "Predicted"])
plt.show()