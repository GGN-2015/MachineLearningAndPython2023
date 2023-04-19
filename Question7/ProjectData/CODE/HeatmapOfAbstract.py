from SampleRateChecker import getRootFilePath
import os, json

import numpy             as np
import seaborn           as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    rootDir         = getRootFilePath()
    bestFeatureFile = os.path.join(rootDir, "DATA/MFCC_ABSTRACT_BEST_FEATURE_ID_AND_PVALUE.json")
    bestFeatureList = json.load(open(bestFeatureFile))
    arrayList       = np.array([0] * 13 * 4).reshape((4, 13))
    for i, (indx, pvalue) in enumerate(bestFeatureList):
        row = indx // 4
        pos = indx  % 4
        arrayList[pos][row] = - np.log(pvalue) / np.log(np.exp(1))
    plt.figure(figsize=(12, 6))
    sns.heatmap(arrayList, cmap="YlOrRd")
    plt.ylabel("Reduce Method")
    plt.xlabel("Frequency ID")
    plt.show()