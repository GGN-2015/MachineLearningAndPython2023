from sklearn.datasets        import make_classification
from sklearn.metrics         import confusion_matrix
from scipy                   import stats
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn                 import svm
from math                    import sqrt

import os
import matplotlib.pyplot as plt
import seaborn           as sns
import pandas            as pd
import numpy             as np

ROOT_PATH = os.path.dirname(__file__)
MAX_FEATURE_CNT = 100
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

# sn, sp, acc, avc, mcc = fCalculateFromConfusionMatrix(tn, fp, fn, tp)
def fCalculateFromConfusionMatrix(tn: float, fp: float, fn: float, tp: float):
    sn  = tp / (tp + fn)
    sp  = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    avc = (sn + sp) / 2
    mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return sn, sp, acc, avc, mcc

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

# svmAcc = AccOnSvm(xData, yData)
def AccOnSvm(xData, yData):
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=TEST_RATE,
                                                    random_state=42,
                                                    stratify=yData)
    tn, fp, fn, tp        = fRunSvm(xTrain, yTrain, xTest, yTest)
    sn, sp, acc, avc, mcc = fCalculateFromConfusionMatrix(tn, fp, fn, tp)
    return acc

# nbayesAcc = AccOnNBayesm(xData, yData)
def AccOnNBayesm(xData, yData):
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=TEST_RATE,
                                                    random_state=42,
                                                    stratify=yData)
    tn, fp, fn, tp        = fRunNbayes(xTrain, yTrain, xTest, yTest)
    sn, sp, acc, avc, mcc = fCalculateFromConfusionMatrix(tn, fp, fn, tp)
    return acc

# knnAcc = AccOnKnn(xData, yData)
def AccOnKnn(xData, yData):
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=TEST_RATE,
                                                    random_state=42,
                                                    stratify=yData)
    tn, fp, fn, tp        = fRunKNN(xTrain, yTrain, xTest, yTest)
    sn, sp, acc, avc, mcc = fCalculateFromConfusionMatrix(tn, fp, fn, tp)
    return acc

if __name__ == "__main__":
    filename = os.path.join(ROOT_PATH, "ALL3.txt")
    fIdAndPList = fGetTopFeatures(filename, False)
    fIdList     = [x for x, _ in fIdAndPList]
    (tSample, tClass, tFeature, tMatrix) = fLoadDataMatrix(filename)
    tMatrix = tMatrix.T
    datas = [[], [], []] # SVM, NBayes, KNN
    for fcnt in range(1, MAX_FEATURE_CNT + 1):
        xData     = tMatrix[:,np.array(fIdList[:fcnt])]
        yData     = (tClass == "POS").astype(int)
        svmAcc    = AccOnSvm    (xData, yData)
        nbayesAcc = AccOnNBayesm(xData, yData)
        knnAcc    = AccOnKnn    (xData, yData)
        datas[0].append(   svmAcc)
        datas[1].append(nbayesAcc)
        datas[2].append(   knnAcc)
    plt.figure(figsize=(19, 4))
    sns.heatmap(
        datas, cmap="YlOrRd", 
        yticklabels=["SVM", "NBayes", "KNN"], 
        xticklabels=list(range(1, MAX_FEATURE_CNT + 1)))
    plt.xlabel("top-N features")
    plt.ylabel("type of classifier")
    plt.title("heatmap for acc of top-N features classifier")
    plt.show()
