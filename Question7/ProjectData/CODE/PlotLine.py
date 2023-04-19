from SampleRateChecker import getRootFilePath

import os
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator




if __name__ == "__main__" :
    rootDir = getRootFilePath()
    tTime = []
    tSn  = []
    tSp  = []
    tAcc = []
    tAVC = []
    tMcc = []
    dataFile = os.path.join(rootDir, "DATA/MFCC_ALPHA_ABSTRACT_SELECTED_PREFIX.json")
    with open(dataFile, "r") as f:
        data_dict = json.load(f)

        for key in data_dict.items():
            tTime.append(key[0])
            tSn.append(key[1]["sn"])
            tSp.append(key[1]["sp"])
            tAcc.append(key[1]["acc"])
            tAVC.append(key[1]["avc"])
            tMcc.append(key[1]["mcc"])
    # plot
    plt.figure(figsize=(15,5))
    plt.plot(tTime, tSn, c='blue', label = "sn")
    plt.plot(tTime, tSp, c='red', label = "sp")
    plt.plot(tTime, tAcc, c='yellow', label = "acc")
    plt.plot(tTime, tAVC, c='green', label = "AVC")
    plt.plot(tTime, tMcc, c='brown', label = "mcc")
    plt.legend(loc="lower right")
    plt.title("Result of the Top n features")
    plt.xlabel("top n features")
    plt.ylabel("result values")
    plt.show()

