import os
from pydub import AudioSegment

# 获取项目数据部分根目录
# rootDir = getRootFilePath()
def getRootFilePath():
    dirname = os.path.dirname(os.path.dirname(__file__))
    return dirname

# 检查一个音频文件并返回其采样率
# sampleRate = getMp3sampleRate(filename)
def getMp3sampleRate(filename):
    audio = AudioSegment.from_file(filename)
    sampleRate = audio.frame_rate
    return sampleRate

# 获取一个目录下的文件名列表
# fileNameList = getFileNameListInDir(dirname)
def getFileNameListInDir(dirname):
    ans = []
    for filename in os.listdir(dirname):
        if os.path.isfile(os.path.join(dirname, filename)):
            ans.append(filename)
    return ans

# 此函数没有复用性
def runCode():
    rootDir        = getRootFilePath()
    Mp3FileDir     = os.path.join(rootDir, "MP3")
    DataFileDir    = os.path.join(rootDir, "DATA")
    outputFileName = os.path.join(DataFileDir, "SAMPLE_RATE.txt")
    fp = open(outputFileName, "w")
    for filename in getFileNameListInDir(Mp3FileDir):
        filepath = os.path.join(Mp3FileDir, filename)
        sampleRate = getMp3sampleRate(filepath)
        print(filename, sampleRate)
        fp.write("%20s %10d\n" % (filename, sampleRate))
    fp.close()

if __name__ == "__main__":
    runCode()