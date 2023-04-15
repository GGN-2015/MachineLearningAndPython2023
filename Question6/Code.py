from sklearn                 import svm
from sklearn.datasets        import make_classification
from sklearn.linear_model    import LassoCV
from sklearn.metrics         import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes     import GaussianNB
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier

import numpy as np
from math import sqrt
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os

REPEAT    = 10
TEST_RATE = 0.3

# check the size of data block
def __fCheckDataAvailable(xTrain, yTrain, xTest, yTest):
    assert type(xTrain) == np.ndarray and len(xTrain.shape) == 2
    assert type(yTrain) == np.ndarray
    trainSetSiz, trainFeatureCnt = xTrain.shape
    assert trainSetSiz == len(yTrain)
    assert type(xTest)  == np.ndarray and len(xTrain.shape) == 2
    assert type(yTest)  == np.ndarray
    testSetSiz, testFeatureCnt = xTest.shape
    assert testSetSiz == len(yTest)
    assert trainFeatureCnt == testFeatureCnt

# tn, fp, fn, tp = fRunSvm(xTrain, yTrain, xTest, yTest)
def fRunSvm(xTrain: np.ndarray, yTrain: np.ndarray, xTest: np.ndarray, yTest: np.ndarray):
    __fCheckDataAvailable(xTrain, yTrain, xTest, yTest)
    # run svm with default argv
    # in yTrain and yTest, we use 0 as NEG, 1 as POS
    assert sum(yTrain == 1) + sum(yTrain == 0) == len(yTrain)
    clf = svm.SVC(kernel="rbf", 
                  class_weight={
                    0: sum(yTrain == 1),
                    1: sum(yTrain == 0)
                })
    clf.fit(xTrain, yTrain)
    yPred = clf.predict(xTest)
    tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
    return (tn, fp, fn, tp)

# tn, fp, fn, tp = fRunNbayes(xTrain, yTrain, xTest, yTest)
def fRunNbayes(xTrain: np.ndarray, yTrain: np.ndarray, xTest: np.ndarray, yTest: np.ndarray):
    __fCheckDataAvailable(xTrain, yTrain, xTest, yTest)
    clf = GaussianNB()
    clf.fit(xTrain, yTrain)
    yPred = clf.predict(xTest)
    tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
    return (tn, fp, fn, tp)

# tn, fp, fn, tp = fRunKNN(xTrain, yTrain, xTest, yTest)
def fRunKNN(xTrain: np.ndarray, yTrain: np.ndarray, xTest: np.ndarray, yTest: np.ndarray):
    __fCheckDataAvailable(xTrain, yTrain, xTest, yTest)
    # run knn with default argv
    clf = KNeighborsClassifier()
    clf.fit(xTrain, yTrain)
    yPred = clf.predict(xTest)
    tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
    return (tn, fp, fn, tp)

# tn, fp, fn, tp = fRunDTree(xTrain, yTrain, xTest, yTest)
def fRunDTree(xTrain: np.ndarray, yTrain: np.ndarray, xTest: np.ndarray, yTest: np.ndarray):
    __fCheckDataAvailable(xTrain, yTrain, xTest, yTest)
    # run knn with default argv
    clf = DecisionTreeClassifier()
    clf.fit(xTrain, yTrain)
    yPred = clf.predict(xTest)
    tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
    return (tn, fp, fn, tp)

# tn, fp, fn, tp = fRunLasso(xTrain, yTrain, xTest, yTest)
def fRunLasso(xTrain: np.ndarray, yTrain: np.ndarray, xTest: np.ndarray, yTest: np.ndarray):
    __fCheckDataAvailable(xTrain, yTrain, xTest, yTest)
    # run knn with default argv
    clf = LassoCV(cv=3)
    clf.fit(xTrain, yTrain)
    yPred = clf.predict(xTest)
    yPred = (yPred > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
    return (tn, fp, fn, tp)

# sn, sp, acc, avc, mcc = fCalculateFromConfusionMatrix(tn, fp, fn, tp)
def fCalculateFromConfusionMatrix(tn: float, fp: float, fn: float, tp: float):
    sn  = tp / (tp + fn)
    sp  = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    avc = (sn + sp) / 2
    mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return sn, sp, acc, avc, mcc

RUN_FUNCTION_LIST = [fRunSvm, fRunNbayes, fRunKNN, fRunDTree, fRunLasso]

# If the random seed is different, it's difficult to compare 
# the training accuracy of different models.
def __fGetRandomStateByRoundId(rid: int):
    assert type(rid) == int
    assert rid >= 0
    # check whether a integer is prime
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True
    # use the nth prime as the random state
    def nth_prime_number(i):
        count = 0
        n = 1
        while count < i:
            n += 1
            if is_prime(n):
                count += 1
        return n
    return nth_prime_number(rid + 1)

DATA_NAME_LIST = ["sn", "sp", "acc", "avc", "mcc"]

def avg(lis: list):
    assert type(lis) == list
    # lis = [x for x in lis if not isnan(x)]
    return sum(lis) / len(lis)

def std(lis: list):
    assert type(lis) == list
    n    = len(lis)
    mean = sum(lis) / n
    var  = sum([(x - mean) ** 2 for x in lis]) / n
    std  = sqrt(var)
    return std

# (sn, sp, acc, avc, mcc), (snStd, spStd, accStd, avcStd, mccStd) = fRunAverageAndStd(xData, yData, trainTestRate, roundCnt)
def fRunAverageAndStd(fRunFunc, xData: np.ndarray, yData: np.ndarray, testRate: float, roundCnt: int):
    assert fRunFunc in RUN_FUNCTION_LIST
    assert type(roundCnt) == int and roundCnt >= 1
    assert 0 < testRate and testRate < 1 # testRate = testSize / (trainSize + testSize)
    assert type(xData) == np.ndarray and len(xData.shape) == 2
    assert type(yData) == np.ndarray and len(yData.shape) == 1
    assert len(xData) == len(yData)
    dataSize  = len(xData)
    testSize  = round(dataSize * testRate)
    trainSize = dataSize - testSize
    assert trainSize > 0 and testSize > 0 # otherwise div 0 will ocurr in the calc step
    (snArr, spArr, accArr, avcArr, mccArr) = [[], [], [], [], []]
    assert snArr is not spArr # make sure the id is different
    for rid in range(roundCnt):
        xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=testRate,
                                                        random_state=__fGetRandomStateByRoundId(rid),
                                                        stratify=yData)
        # core function call: fRunFunc is a preict model
        tn, fp, fn, tp = fRunFunc(xTrain, yTrain, xTest, yTest)
        sn, sp, acc, avc, mcc = fCalculateFromConfusionMatrix(tn, fp, fn, tp)
        snArr .append(sn )
        spArr .append(sp )
        accArr.append(acc)
        avcArr.append(avc)
        mccArr.append(mcc)
    ansAvg = [avg(snArr), avg(spArr), avg(accArr), avg(avcArr), avg(mccArr)]
    ansStd = [std(snArr), std(spArr), std(accArr), std(avcArr), std(mccArr)]
    return ansAvg, ansStd

def __fGetFuncName(func):
    return str(func).split()[1] # get function name

def __fGetColorNameById(id: int):
    COLORS = ["red", "orange", "green", "blue", "purple"]
    assert 0 <= id and id < len(COLORS)
    return COLORS[id]

# there are 25 bars: len(RUN_FUNCTION_LIST) * len([sn, sp, acc, avc, mcc])
# Plt is usally a subplot
def fPlot25bars(Plt, statData: dict):
    assert type(statData) == dict
    assert len(RUN_FUNCTION_LIST) == 5
    assert len(DATA_NAME_LIST)    == 5
    for func in RUN_FUNCTION_LIST:
        assert statData.get(__fGetFuncName(func)) is not None
        for dataName in DATA_NAME_LIST:
            assert statData[__fGetFuncName(func)].get(dataName) is not None
    datas    = [[], [], [], [], []]
    dataStds = [[], [], [], [], []]
    for i in range(5):
        for j in range(5):
            datas[i].append(
                statData[__fGetFuncName(RUN_FUNCTION_LIST[i])][DATA_NAME_LIST[j]])
            dataStds[i].append(
                statData[__fGetFuncName(RUN_FUNCTION_LIST[i])][DATA_NAME_LIST[j] + "Std"])
        datas[i] = np.array(datas[i])
    for i in range(5):
        Plt.bar(np.arange(5) * 6 + i, datas[i], yerr=dataStds[i],
                color=__fGetColorNameById(i), 
                label=__fGetFuncName(RUN_FUNCTION_LIST[i])[4:])
    Plt.legend()
    Plt.set_xticks(np.arange(7) * 6 + 2)
    Plt.set_xticklabels(DATA_NAME_LIST + [""] * 2)

# show a plot in axs(subplot)
def fRunOnSingleData(axs, xData, yData) -> None:
    assert type(xData) == np.ndarray and len(xData.shape) == 2
    assert type(yData) == np.ndarray
    assert xData.shape[0] == len(yData)
    def fTestRun(fRunFunc, xData, yData) -> None:
        assert fRunFunc in RUN_FUNCTION_LIST
        (sn, sp, acc, avc, mcc), (snStd, spStd, accStd, avcStd, mccStd) = fRunAverageAndStd(fRunFunc, 
                                                                                            xData, yData, TEST_RATE, REPEAT)
        return {
            "func": __fGetFuncName(fRunFunc), 
            "sn": sn, "sp": sp, "acc": acc, "avc": avc, "mcc": mcc,
            "snStd" :  snStd,
            "spStd" :  spStd,
            "accStd": accStd,
            "avcStd": avcStd,
            "mccStd": mccStd
        }
    statData = {}
    for fRunFunc in RUN_FUNCTION_LIST:
       tmpData = fTestRun(fRunFunc, xData, yData)
       statData[__fGetFuncName(fRunFunc)] = tmpData
    fPlot25bars(axs, statData)

# this function is only used for test
def fTestPlt():
    fig, axs = plt.subplots()
    xData, yData = make_classification(n_samples=100, n_features=3, 
                                       n_informative=2, n_redundant=0, random_state=42)
    fRunOnSingleData(axs, xData, yData)
    plt.show()

# (Sample, Class, Feature, Matrix) = fLoadDataMatrix(filename)
def fLoadDataMatrix(filename: str):
    assert type(filename) == str
    data = pd.read_table(filename, delimiter="\t", 
                        header=None, skiprows=[0])
    tMatrix  = data.iloc[:, 1:].values
    tFeature = (pd.read_table(filename, delimiter="\t", header=None, usecols=[0]).iloc[1:, :].values.T)[0]
    tClass   = pd.read_table(filename, delimiter="\t", header=None, nrows=1).values[0][1:]
    tSample  = np.array(list(range((tMatrix.shape)[1])))
    return (tSample, tClass, tFeature, tMatrix)

# (posSet, negSet) = fSplitPosAndNeg(tMatrix, tClass)
def fSplitPosAndNeg(tMatrix, tClass):
    assert type(tMatrix) == np.ndarray
    assert type(tClass) == np.ndarray
    return (
        tMatrix[:, tClass == "POS"],
        tMatrix[:, tClass == "NEG"]
    )

# (tValue, pValue) = fT_test(posSet, negSet)
def fT_test(posSet, negSet):
    tValues, pValues = stats.ttest_ind(posSet, negSet, axis=1, equal_var=True, nan_policy='propagate')
    return tValues, pValues

# (featureId, pValue) = topFeatureId = fTopFeatureId(tValue)
def fTopFeatureId(pValue):
    tValueSorted = list(enumerate(pValue))
    tValueSorted.sort(key=lambda x:x[1])
    return tValueSorted

# return a list of (featureId, pValue)
def fGetTopFeatures(filename: str, showList=False):
    assert type(filename) == str
    tSample, tClass, tFeature, tMatrix = fLoadDataMatrix(filename)
    posSet, negSet = fSplitPosAndNeg(tMatrix, tClass)
    tValues, pValues = fT_test(posSet, negSet)
    topFeatureId = fTopFeatureId(pValues)
    if showList:
        for (rank, (featureId, pValue)) in enumerate(topFeatureId):
            print("rank = %6d, featureId = %6d, featureName = %25s, pValue = %.6f, tValue = %10.6f" % (rank + 1, featureId, tFeature[featureId], pValue, tValues[featureId]))
    return topFeatureId

# the following function is not designed for generalize
def fDrawByFeatureIdSet(axs, featureSortedList, lpos, rpos, tClass, tMatrix):
    featureSortedList = map(lambda x: x[0], featureSortedList[lpos : rpos])
    xData = [tMatrix[id] for id in featureSortedList]
    xData = (np.array(xData)).T
    yData = (tClass == "POS").astype(int)
    fRunOnSingleData(axs, xData, yData)

# draw top-1, top-10, top-100, end-100
def fMainPlotFunction(filename):
    tSample, tClass, tFeature, tMatrix = fLoadDataMatrix(filename)
    topFeatureId = fGetTopFeatures(filename)
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(15, 12))
    featureCnt = len(tFeature)
    L     = [0,  0,   0, featureCnt - 100]
    R     = [1, 10, 100, featureCnt      ]
    TITLE = ["Top-1", "Top-10", "Top-100", "End-100"]
    for i in range(4):
        fDrawByFeatureIdSet(axs[i], topFeatureId, L[i], R[i], tClass, tMatrix)
        axs[i].set_title(TITLE[i])
    figName = "Ttest Based Training"
    # plt.gcf().canvas.set_window_title(figName)
    fig.suptitle(figName)
    plt.show()

if __name__ == "__main__":
    PATHNAME = os.path.dirname(__file__)
    FILENAME = PATHNAME + "/" +"ALL3.txt"
    fMainPlotFunction(FILENAME)
