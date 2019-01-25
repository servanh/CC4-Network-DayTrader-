import numpy as np
import collections

#Float to unary function
def floatToUnary(floatdata, granularity):
    #Move the lowest absolute value into the ones place
    minVal1 = min(np.vectorize(abs)(floatdata))
    floatdata = floatdata/minVal1
    #Make them all positive
    minVal2 = min(floatdata)
    floatdata = floatdata - minVal2
    #Divide the data range by the granularity
    dataRange = max(floatdata)
    floatdata = list(map(lambda fd: int(granularity*fd/dataRange), floatdata))
    #Convert to unary
    unaryData = list(map(lambda fd: np.array(list(''.rjust(fd, '1').rjust(granularity, '0')), dtype=int), floatdata))
    #return both the unaryData and the information to convert it back to a float
    return ([dataRange, minVal1, minVal2, granularity], unaryData)

def unaryToFloat(unaryData, floatInfo):
    (dataRange, minVal1, minVal2, granularity) = floatInfo
    floatdata = list(map(lambda ud: (dataRange*collections.Counter(ud)[1]/granularity+minVal2)*minVal1, unaryData))
    return floatdata