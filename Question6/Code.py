from sklearn                 import svm
from sklearn.datasets        import make_classification
from sklearn.metrics         import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes     import GaussianNB
from sklearn.neighbors       import KNeighborsClassifier

import numpy as np
from math import sqrt

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
    clf = svm.SVC(kernel="rbf")
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

# sn, sp, acc, avc, mcc = fCalculateFromConfusionMatrix(tn, fp, fn, tp)
def fCalculateFromConfusionMatrix(tn: float, fp: float, fn: float, tp: float):
    sn  = tp / (tp + fn)
    sp  = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    avc = (sn + sp) / 2
    mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return sn, sp, acc, avc, mcc

RUN_FUNCTION_LIST = [fRunSvm, fRunNbayes, fRunKNN]

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

# sn, sp, acc, avc, mcc = fRunAverageSvm(xData, yData, trainTestRate, roundCnt)
def fRunAverage(fRunFunc, xData: np.ndarray, yData: np.ndarray, testRate: float, roundCnt: int):
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
    (snArr, spArr, accArr, avcArr, mccArr) = ([], [], [], [], [])
    assert snArr is not spArr # make sure the id is different
    for rid in range(roundCnt):
        xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=testRate
                                                        random_state=__fGetRandomStateByRoundId(rid))
        # core function call: fRunFunc is a preict model
        tn, fp, fn, tp = fRunFunc(xTrain, yTrain, xTest, yTest)
        sn, sp, acc, avc, mcc = fCalculateFromConfusionMatrix(tn, fp, fn, tp)
        snArr .append(sn )
        spArr .append(sp )
        accArr.append(acc)
        avcArr.append(avc)
        mccArr.append(mcc)
    ansSum = [sum(snArr), sum(spArr), sum(accArr), sum(avcArr), sum(mccArr)]
    ansAvg = [x / roundCnt for x in ansSum]
    return ansAvg

# this function is only used for testing
def fTestRun(fRunFunc) -> None:
    assert fRunFunc in RUN_FUNCTION_LIST
    xData, yData = make_classification(n_samples=100, n_features=3, 
                                       n_informative=2, n_redundant=0, random_state=42)
    sn, sp, acc, avc, mcc = fRunAverage(fRunFunc, xData, yData, 0.3, 10)
    print({
        "sn": sn, "sp": sp, "acc": acc, "avc": avc, "mcc": mcc
    })

if __name__ == "__main__":
    fTestRun(fRunKNN)