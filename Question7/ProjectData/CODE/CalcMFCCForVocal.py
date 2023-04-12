import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

from SampleRateChecker import getRootFilePath, getFileNameListInDir

# mfccs = solveMFCC(filename)
# 计算给定音频片段的梅尔倒谱
def solveMFCC(filename: set):
    y, sr = librosa.load(filename)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs

# 计算音乐的分贝数
def getDb(matrix):
    return 10 * np.log(matrix + 1) / np.log(10)

# 显示一段音频的梅尔倒谱
def debugShow(filename: set):
    mfccs    = solveMFCC(filename)[:12,:]
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rootDir   = getRootFilePath()
    segPosDir = os.path.join(rootDir, "VOCALSEG/POS")
    segNegDir = os.path.join(rootDir, "VOCALSEG/NEG")
    dataFile  = os.path.join(rootDir, "DATA/MFCC_VOCAL_ALPHA.txt")
    fp = open(dataFile, "w")
    for filename in getFileNameListInDir(segPosDir):
        print("filename = %s" % filename)
        fileNow = os.path.join(segPosDir, filename)
        mfccs = solveMFCC(fileNow)
        assert mfccs.shape == (13, 87) # 确保所有矩阵形状相同
        fp.write("%s " % filename)
        fp.write("POS ")
        for i in range(mfccs.shape[0]):
            for j in range(mfccs.shape[1]):
                fp.write("%15.6lf " % mfccs[i, j])
        fp.write("\n")
    for filename in getFileNameListInDir(segNegDir):
        print("filename = %s" % filename)
        fileNow = os.path.join(segNegDir, filename)
        mfccs = solveMFCC(fileNow)
        assert mfccs.shape == (13, 87) # 确保所有矩阵形状相同
        fp.write("%s " % filename)
        fp.write("NEG ")
        for i in range(mfccs.shape[0]):
            for j in range(mfccs.shape[1]):
                fp.write("%15.6lf " % mfccs[i, j])
        fp.write("\n")
    print("Done.")
