import numpy as np

from SampleRateChecker import getRootFilePath
import os

# tSample, tClass, tFeatures = dataLoader(filename)
def dataLoader(filename):
    tSample  = []
    tClass   = []
    tFeature = []
    for line in open(filename):
        if line.strip() == "": continue
        tData = line.split()
        tSample .append(tData[0  ])
        tClass  .append(tData[1  ])
        tFeature.append(list(map(float, tData[2: ])))
    return (
        np.array(tSample), 
        np.array(tClass), 
        np.array(tFeature, dtype=float)
    )

if __name__ == "__main__":
    rootDir  = getRootFilePath()
    dataFile = os.path.join(rootDir, "DATA/MFCC_ALPHA.txt")
    tSample, tClass, tFeatures = dataLoader(dataFile)