import numpy as np
from sklearn.metrics         import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm             import SVC

from SampleRateChecker import getRootFilePath
import os
from math import sqrt
import json

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

# 使用线性核函数的支持向量机进行测试
# tn, fp, fn, tp = testSVC(tSample, tClass, tFeatures, 0.2)
def testSVC(tSample, tClass, tFeatures, testSize: float):
    assert type(tSample  ) == np.ndarray
    assert type(tClass   ) == np.ndarray
    assert type(tFeatures) == np.ndarray
    xData       = tFeatures
    yData       = (tClass == "POS").astype(int)
    classWeight = {
        0: np.sum(yData), 
        1: len(yData) - np.sum(yData)
    }
    xTrain, xTest, yTrain, yTest = train_test_split(
        xData, 
        yData, 
        test_size=testSize, 
        stratify=yData,
        random_state=42
    )
    svc = SVC(kernel="rbf", class_weight=classWeight)
    svc.fit(xTrain, yTrain)
    yPred = svc.predict(xTest)
    tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
    return (tn, fp, fn, tp)

# 根据数据获得支持向量机
# svm = getSvcOnData(tClass, tFeatures)
def getSvcOnData(tClass, tFeatures):
    assert type(tClass   ) == np.ndarray
    assert type(tFeatures) == np.ndarray
    xData       = tFeatures
    yData       = (tClass == "POS").astype(int)
    classWeight = {
        0: np.sum(yData), 
        1: len(yData) - np.sum(yData)
    }
    svc = SVC(kernel="rbf", class_weight=classWeight)
    svc.fit(xData, yData)
    return svc

# 使用线性核函数的支持向量机进行测试
# tn, fp, fn, tp = testSVC(tSample, tClass, tFeatures)
def testSVCFull(tSample, tClass, tFeatures):
    assert type(tSample  ) == np.ndarray
    assert type(tClass   ) == np.ndarray
    assert type(tFeatures) == np.ndarray
    xData       = tFeatures
    yData       = (tClass == "POS").astype(int)
    classWeight = {
        0: np.sum(yData), 
        1: len(yData) - np.sum(yData)
    }
    svc = SVC(kernel="rbf", class_weight=classWeight)
    svc.fit(xData, yData)
    yPred = svc.predict(xData)
    tn, fp, fn, tp = confusion_matrix(yData, yPred).ravel()
    return (tn, fp, fn, tp)

# sn, sp, acc, avc, mcc = calculateFromConfusionMatrix(tn, fp, fn, tp)
def calculateFromConfusionMatrix(tn: float, fp: float, fn: float, tp: float):
    sn  = tp / (tp + fn)
    sp  = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    avc = (sn + sp) / 2
    mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return sn, sp, acc, avc, mcc

# 在某个数据文件上测试 SVC 的效果，可复用
def testSVConFile(filename):
    tSample, tClass, tFeatures = dataLoader(filename)
    tn, fp, fn, tp = testSVC(tSample, tClass, tFeatures, 0.2)
    sn, sp, acc, avc, mcc = calculateFromConfusionMatrix(tn, fp, fn, tp)
    print({
        "sn" :  sn,
        "sp" :  sp,
        "acc": acc,
        "avc": avc,
        "mcc": mcc
    })

# 不可复用模块
def testSVConMFCC_ALPHA():
    rootDir  = getRootFilePath()
    dataFile = os.path.join(rootDir, "DATA/MFCC_ALPHA.txt")
    outpFile = os.path.join(rootDir, "DATA/TESTRATE_STAT_ALPHA.json")
    tSample, tClass, tFeatures = dataLoader(dataFile)
    dataDict = {}
    for testRate in range(10, 90):
        print("testRate = %2d ..." % testRate)
        tn, fp, fn, tp = testSVC(tSample, tClass, tFeatures, testRate / 100)
        sn, sp, acc, avc, mcc = calculateFromConfusionMatrix(tn, fp, fn, tp)
        dictNow = {
            "sn" :  sn,
            "sp" :  sp,
            "acc": acc,
            "avc": avc,
            "mcc": mcc
        }
        dataDict[testRate] = dictNow
    with open(outpFile, "w") as fp:
        json.dump(dataDict, fp, indent=4)

if __name__ == "__main__":
    rootDir = getRootFilePath()
    filename = os.path.join(rootDir, "DATA/MFCC_VOCAL_ALPHA.txt")
    testSVConFile(filename)