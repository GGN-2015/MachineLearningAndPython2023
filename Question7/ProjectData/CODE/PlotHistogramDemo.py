from DataLoader import dataLoader
from SampleRateChecker import getRootFilePath

import numpy as np
import matplotlib.pyplot as plt
import math
import os

BINS_COUNT = 100
ALPHA = 0.3

if __name__ == "__main__":
    rootDir  = getRootFilePath()
    dataFile = os.path.join(rootDir, "DATA/MFCC_ALPHA.txt") 
    arrPos = []                 # Pos 
    arrNeg = []                 # Neg
    arrAll = []                 # All
    tSample, tClass, tFeatures = dataLoader(dataFile)
    for indx in range(len(tClass)):
        dataNow = tFeatures[indx].reshape((13, 87))
        maxNow  = np.mean(dataNow[0, :])
        if tClass[indx] == "POS":
            arrPos.append(maxNow)
        else:
            arrNeg.append(maxNow)
        arrAll.append(maxNow)
    npPos = np.array(arrPos)
    npNeg = np.array(arrNeg)
    npAll = np.array(arrAll)
    maxValue = math.floor(np.max(npAll)) + 1
    minValue = math.floor(np.min(npAll)) 
    difference = maxValue - minValue   # 58
    print("maxvalue: %d" % maxValue + " minValue: %d" % minValue)
    print("difference: %d" % difference)
    plt.figure(figsize=(12, 3))
    plt.hist(npPos, bins=BINS_COUNT, alpha=ALPHA, label="POS", color="red")
    plt.hist(npNeg, bins=BINS_COUNT, alpha=ALPHA, label="NEG", color="blue")
    plt.hist(npAll, bins=BINS_COUNT, alpha=ALPHA, label="ALL", color="orange")
    plt.legend(loc="upper right")
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Amplitude from MFCC')
    plt.show()