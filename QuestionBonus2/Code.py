import pandas as pd
import numpy as np
from scipy import stats
from math import sqrt

import os
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
DIRNAME = os.path.dirname(__file__)

REPEAT    = 10
TEST_RATE = 0.3

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

# topFeatureId = fTopFeatureId(tValue)
def fTopFeatureId(pValue, topN = 10):
    tValueSorted = list(enumerate(pValue))
    tValueSorted.sort(key=lambda x:x[1])
    return tValueSorted[:topN]

# topFeatureId = fGetTopFeaturesInDataFile(filename)
def fGetTopFeaturesInDataFile(filename: str) -> list:
    assert type(filename) == str
    tSample, tClass, tFeature, tMatrix = fLoadDataMatrix(filename)
    posSet, negSet = fSplitPosAndNeg(tMatrix, tClass)
    tValues, pValues = fT_test(posSet, negSet)
    topFeatureId = fTopFeatureId(pValues, topN = len(tFeature))
    ans = []
    for featureId, pValue in topFeatureId:
        ans.append(featureId)
    return ans

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

# run svm 10 random class
def runSvmOnFeatureSet(featureSet, tMatrix, tClass):
    xData = tMatrix[featureSet, :]
    yData = tClass
    for rid in range(REPEAT):
        xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=TEST_RATE,
                                                        random_state=__fGetRandomStateByRoundId(rid),
                                                        stratify=yData)
        # TODO

# sn, sp, acc, avc, mcc = fCalculateFromConfusionMatrix(tn, fp, fn, tp)
def fCalculateFromConfusionMatrix(tn: float, fp: float, fn: float, tp: float):
    sn  = tp / (tp + fn)
    sp  = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    avc = (sn + sp) / 2
    mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return sn, sp, acc, avc, mcc

if __name__ == "__main__":
    filename = os.path.join(DIRNAME, "ALL3.txt")
    topFeatureId = fGetTopFeaturesInDataFile(filename)