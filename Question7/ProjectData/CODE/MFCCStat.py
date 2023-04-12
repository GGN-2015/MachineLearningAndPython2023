import numpy as np
import os

from DataLoader              import dataLoader, testSVC, calculateFromConfusionMatrix
from SampleRateChecker       import getRootFilePath
from sklearn.model_selection import train_test_split

# 从数据矩阵中获取每行的最小值最大值均值和标准差
# abstractMatrix = getMinMaxMeanStd(tFeatures)
def getMinMaxMeanStd(tFeatures: np.ndarray):
    tMin  = np.min (tFeatures, axis=1)
    tMax  = np.max (tFeatures, axis=1)
    tMean = np.mean(tFeatures, axis=1)
    tStd  = np.var (tFeatures, axis=1) ** 0.5
    return np.c_[tMin, tMax, tMean, tStd]

# 从文件读取数据并进行抽象化
# tSample, tClass, abstractData = getAbstractDataFromFile(filename)
def getAbstractDataFromFile(filename):
    tSample, tClass, tFeatures = dataLoader(filename)
    assert tFeatures.shape[1] == 13 * 87
    abstFeatures = []
    for i in range(len(tFeatures)):
        rawData         = tFeatures[i].reshape((13, 87))
        abstractRawData = getMinMaxMeanStd(rawData)
        abstFeatures.append(abstractRawData.reshape((-1, 1)).squeeze())
    return tSample, tClass, np.array(abstFeatures)

# 此函数不考虑复用性
def createAbstractDataFile():
    rootDir    = getRootFilePath()
    dataFile   = os.path.join(rootDir, "DATA/MFCC_ALPHA.txt")
    outputFile = os.path.join(rootDir, "DATA/MFCC_ALPHA_ABSTRACT.txt")
    tSample, tClass, abstractData = getAbstractDataFromFile(dataFile)
    fp = open(outputFile, "w")
    for indx in range(len(tSample)):
        fp.write("%20s %4s " % (tSample[indx], tClass[indx]))
        for j in range(abstractData.shape[1]):
            fp.write("%15f " % abstractData[indx, j])
        fp.write("\n")
    fp.close()

# 此函数不考虑复用性
def testSvmOnAbstractData(testSize: float):
    rootDir  = getRootFilePath()
    dataFile = os.path.join(rootDir, "DATA/MFCC_ALPHA.txt")
    tSample, tClass, abstractData = getAbstractDataFromFile(dataFile)
    tn, fp, fn, tp = testSVC(tSample, tClass, abstractData, testSize)
    sn, sp, acc, avc, mcc = calculateFromConfusionMatrix(tn, fp, fn, tp)
    print({
        "sn" :  sn,
        "sp" :  sp,
        "acc": acc,
        "avc": avc,
        "mcc": mcc
    })

if __name__ == "__main__":
    createAbstractDataFile()