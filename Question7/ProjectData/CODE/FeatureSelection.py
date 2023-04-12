from DataLoader        import dataLoader
from SampleRateChecker import getRootFilePath
from scipy             import stats
import numpy as np
import os
import json

# (posSet, negSet) = splitPosAndNeg(tMatrix, tClass)
def splitPosAndNeg(tMatrix, tClass):
    assert type(tMatrix) == np.ndarray
    assert type(tClass) == np.ndarray
    return (
        tMatrix[tClass == "POS", :],
        tMatrix[tClass == "NEG", :]
    )

# (tValue, pValue) = ttest(posSet, negSet)
def ttest(posSet, negSet):
    tValues, pValues = stats.ttest_ind(posSet.T, negSet.T, axis=1, equal_var=True, nan_policy='propagate')
    return tValues, pValues

# topFeatureId = topFeatureId(tValue)
def getTopFeatureId(pValue, topN = 10):
    tValueSorted = list(enumerate(pValue))
    tValueSorted.sort(key=lambda x:x[1])
    return tValueSorted[:topN]

# featureIdList = showTopFeaturesInDataFile(filename)
def showTopFeaturesInDataFile(filename: str):
    assert type(filename) == str
    tSample, tClass, tMatrix = dataLoader(filename)
    posSet, negSet = splitPosAndNeg(tMatrix, tClass)
    tValues, pValues = ttest(posSet, negSet)
    topFeatureId = getTopFeatureId(pValues, topN = 52)
    featureIdList = []
    for featureId, pValue in topFeatureId:
        print("featureId = %6d, pValue = %.30f, tValue = %10.6f" % (featureId, pValue, tValues[featureId]))
        featureIdList.append(featureId)
    return featureIdList

if __name__ == "__main__":
    rtDir         = getRootFilePath()
    dataFile      = os.path.join(rtDir, "DATA/MFCC_ALPHA_ABSTRACT.txt")
    outputFile    = os.path.join(rtDir, "DATA/MFCC_ABSTRACT_BEST_FEATURE_ID.json")
    featureIdList = showTopFeaturesInDataFile(dataFile)
    with open(outputFile, "w") as fp:
        json.dump(featureIdList, fp)
