import os
from pydub import AudioSegment
from SampleRateChecker import getRootFilePath, getMp3sampleRate, getFileNameListInDir

# 目标采样率
TARGET_SAMPLE_RATE=44100

# 修改音频文件的采样率
def changeMp3SampleRate(filename: str, outputFile: str, targetSampleRate: int):
    audio = AudioSegment.from_file(filename, format="mp3")
    audio = audio.set_frame_rate(targetSampleRate)
    audio.export(outputFile, format="mp3")

# 此函数没有复用性
def runCode():
    rootDir        = getRootFilePath()
    Mp3FileDir     = os.path.join(rootDir, "MP3")
    ModMp3FileDir  = os.path.join(rootDir, "MODMP3")
    outputFileName = os.path.join(rootDir, "DATA/SAMPLE_RATE_BETA.txt")
    for filename in getFileNameListInDir(Mp3FileDir): # check sample rate
        inFile  = os.path.join(   Mp3FileDir, filename)
        outFile = os.path.join(ModMp3FileDir, filename)
        print("changeMp3SampleRate for filename = %s ..." % filename)
        changeMp3SampleRate(inFile, outFile, TARGET_SAMPLE_RATE)
    fp = open(outputFileName, "w")
    for filename in getFileNameListInDir(ModMp3FileDir):
        filepath = os.path.join(ModMp3FileDir, filename)
        sampleRate = getMp3sampleRate(filepath)
        print(filename, sampleRate)
        fp.write("%20s %10d\n" % (filename, sampleRate))
    fp.close()

if __name__ == "__main__":
    runCode()