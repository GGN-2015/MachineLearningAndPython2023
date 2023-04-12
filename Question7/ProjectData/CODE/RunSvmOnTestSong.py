from DataLoader        import dataLoader, getSvcOnData
from SampleRateChecker import getRootFilePath
from pydub             import AudioSegment
from CalcMFCC          import solveMFCC

import os
TIME_SPEC = 2 # second, DO NOT CHANGE THIS

# 从文件数据获得 SVM （建议从含 BGM 的数据中获得）
# svm = getSvmFromFile(filename)
def getSvmFromFile(filename):
    _, tClass, tFeatures = dataLoader(filename)
    svm = getSvcOnData(tClass, tFeatures)
    return svm

# 难以复用的模块
# svm = getSvmFromMfccRaw()
def getSvmFromMfccRaw():
    rootDir  = getRootFilePath()
    filename = os.path.join(rootDir, "DATA/MFCC_ALPHA.txt")
    svm = getSvmFromFile(filename)
    return svm

# 读取音频，切分音频，给出间奏判断序列
# tagList = RunSvmOnSong(songFileName)
def RunSvmOnSong(svm, songFileName):
    rootDir     = getRootFilePath()
    audio       = AudioSegment.from_file(songFileName, format="mp3")
    totalLen    = len(audio)
    tmpFileName = os.path.join(rootDir, "TEMPDIR/temp.mp3")
    startTime   = 0
    TIME_SPECMs = TIME_SPEC * 1000
    tagList     = []
    while startTime + TIME_SPECMs < totalLen:
        print("    ----- %10.3lf" % (startTime / totalLen))
        segment = audio[startTime: startTime + TIME_SPECMs]
        segment.export(tmpFileName, format="mp3")
        mfccs = solveMFCC(tmpFileName).reshape((-1,1)).T
        ans   = svm.predict(mfccs)
        tagList.append(ans[0])
        startTime += TIME_SPECMs
    return tagList

# 分析 tagList 中最长连续零的片段，并认为这段片段是我选取的音频片段
# bestPos, bestRecord = tagListZeroAnalysis(tagList)
def tagListZeroAnalysis(tagList: list):
    posNow = -1
    record = {}
    for i in range(len(tagList)):
        dataNow = tagList[i]
        if dataNow == 1:
            posNow = -1
        else: # dataNow = 0
            if posNow == -1: # beginning
                posNow = i
                record[i] = 1
            else: # middle
                record[posNow] += 1
    bestRecord = 0
    bestPos    = -1
    for x in record:
        if record[x] > bestRecord:
            bestRecord = record[x]
            bestPos    = x
    return bestPos, bestRecord

if __name__ == "__main__":
    rootDir  = getRootFilePath()
    test0001 = os.path.join(rootDir, "TESTMP3/TEST_0001.mp3")
    svm      = getSvmFromMfccRaw()
    tagList  = RunSvmOnSong(svm, test0001)
    bestPos, bestRecord = tagListZeroAnalysis(tagList)
    print((bestPos, bestRecord))