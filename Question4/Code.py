import pandas as pd
import numpy as np
from scipy import stats

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
    tValue, pValue = stats.ttest_ind(posSet, negSet, axis=1, equal_var=True, nan_policy='propagate')
    return tValue, pValue

# topFeatureId = fTopFeatureId(tValue)
def fTopFeatureId(tValue, topN = 10):
    tValueSorted = list(enumerate(tValue))
    tValueSorted.sort(key=lambda x:-x[1])
    return tValueSorted[:topN]

if __name__ == "__main__":
    tSample, tClass, tFeature, tMatrix = fLoadDataMatrix("ALL3.txt")
    posSet, negSet = fSplitPosAndNeg(tMatrix, tClass)
    tValue, pValue = fT_test(posSet, negSet)
    topFeatureId = fTopFeatureId(tValue, topN = 10)
    print(topFeatureId)