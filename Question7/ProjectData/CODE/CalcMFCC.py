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

def debugShow(filename: set):
    mfccs = solveMFCC(filename)
    print(mfccs.shape)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rootDir = getRootFilePath()
    segPosDir  = os.path.join(rootDir, "SEG/POS")
    segNegDir  = os.path.join(rootDir, "SEG/NEG")
    for filename in getFileNameListInDir(segPosDir):
        fileNow = os.path.join(segPosDir, filename)
        debugShow(fileNow)
