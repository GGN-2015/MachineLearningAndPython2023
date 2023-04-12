import pandas as pd
import numpy as np
from scipy import stats
import sys
import matplotlib.pyplot as plt

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

# draw a scatter on a subplot(axs)
def fShowScatterOnPlot(plt, featureIdX: int, featureIdY: int, tMatrix: np.ndarray, tClass: np.ndarray, tFeature: np.ndarray):
    assert type(featureIdX) == int
    assert type(featureIdY) == int
    assert type(tMatrix)    == np.ndarray
    assert type(tClass)     == np.ndarray
    assert type(tFeature)   == np.ndarray
    (posSet, negSet) = fSplitPosAndNeg(tMatrix, tClass)
    plt.scatter(negSet[featureIdX], negSet[featureIdY], c="blue", label="NEG")
    plt.scatter(posSet[featureIdX], posSet[featureIdY], c="red" , label="POS")
    plt.set_xlabel(tFeature[featureIdX])
    plt.set_ylabel(tFeature[featureIdY])
    plt.legend()

# this function is only used in the current program
def fMainPlotFunction(filename: str, rankPair: list, figName:str):
    assert type(filename) == str
    assert type(rankPair) == list
    assert len(rankPair) == 2
    assert len(rankPair[0]) == 2 and len(rankPair[1]) == 2
    assert type(figName) == str
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    (tSample, tClass, tFeature, tMatrix) = fLoadDataMatrix(filename)
    tPValueList = fGetTopFeatures(filename)
    for i in range(2):
        for j in range(2):
            rkX, rkY = rankPair[i][j]
            idX, idY = map(lambda x: tPValueList[x - 1][0], rankPair[i][j])
            fShowScatterOnPlot(axs[i, j], idX, idY, tMatrix, tClass, tFeature)
            axs[i, j].set_title("Rank" + str(rkX) + " vs " + "Rank" + str(rkY))
    # plt.gcf().canvas.set_window_title(figName)
    fig.suptitle(figName)
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

if __name__ == "__main__":
    assert len(sys.argv) == 2
    filename = sys.argv[1]
    RANK_PAIR = [[(1, 2), (9, 10)], [(1000, 1001), (10000, 10001)]]
    FIGNAME   = "Dot plots of t-test based"
    fMainPlotFunction(filename, RANK_PAIR, FIGNAME)
