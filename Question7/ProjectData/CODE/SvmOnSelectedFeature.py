import json
from DataLoader        import dataLoader, testSVC, calculateFromConfusionMatrix
from SampleRateChecker import getRootFilePath

import os
import numpy as np

# 获得最优特征列表
# featureIdList = getTopFeatureIdList()
def getTopFeatureIdList():
    rtDir  = getRootFilePath()
    lsFile = os.path.join(rtDir, "DATA/MFCC_ABSTRACT_BEST_FEATURE_ID.json")
    return json.load(open(lsFile))

# 内部使用
# tSample, tClass, tFeature = getDataWithTopNFeature(filename, N)
def getDataWithTopNFeature(filename, N: int):
    tSample, tClass, tFeatures = dataLoader(filename)
    assert tFeatures.shape[1] == 52
    featureIdList = np.array(getTopFeatureIdList()[:N])
    return tSample, tClass, tFeatures[:, featureIdList]

# 获取特征选择，选择了最优 N 个特征后的摘要数据集
# tSample, tClass, tSelectedFeature = getDataWithTopNFeature(filename, N)
def getAbstractDataWithTopNFeature(N: int):
    rtDir     = getRootFilePath()
    filename  = os.path.join(rtDir, "DATA/MFCC_ALPHA_ABSTRACT.txt")
    tSample, tClass, tFeatures = dataLoader(filename)
    assert tFeatures.shape[1] == 52
    featureIdList = np.array(getTopFeatureIdList()[:N])
    return tSample, tClass, tFeatures[:, featureIdList]

# 在含前 N 个最优特征的数据集上测试 SVC
def testSvcOnTopNFeature(N: int, testSize: float):
    tSample, tClass, tSelectedFeature = getAbstractDataWithTopNFeature(N)
    tn, fp, fn, tp = testSVC(tSample, tClass, tSelectedFeature, testSize)
    sn, sp, acc, avc, mcc = calculateFromConfusionMatrix(tn, fp, fn, tp)
    return {
        "sn": sn,
        "sp": sp,
        "acc": acc,
        "avc": avc,
        "mcc": mcc
    }

if __name__ == "__main__":
    dic = {}
    rtDir     = getRootFilePath()
    filename  = os.path.join(rtDir, "DATA/MFCC_ALPHA_ABSTRACT_SELECTED_PREFIX.json")
    for n in range(1, 52 + 1):
        dic[n] = testSvcOnTopNFeature(n, 0.2)
        print(dic[n])
    with open(filename, "w") as fp:
        json.dump(dic, fp, indent=4)
