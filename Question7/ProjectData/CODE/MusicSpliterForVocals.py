from pydub import AudioSegment
import os

from SampleRateChecker import getRootFilePath, getFileNameListInDir
from MusicSpliter import TagFile
TIME_SPEC = 2 # second

# 将音频分割为以 timeSpecMs 为单位长度的片段
def SplitVocalAudioFile(filename, timeSpec: int, rootFolder: str):
    fileId         = filename.split('.')[0]
    vocalWavFolder = os.path.join(rootDir, "VOCALWAV")
    posMp3Folder   = os.path.join(rootFolder, "VOCALSEG/POS")
    negMp3Folder   = os.path.join(rootFolder, "VOCALSEG/NEG")
    tagFolder      = os.path.join(rootFolder, "TAG")
    wavFile        = os.path.join(vocalWavFolder, filename)
    tagFile        = os.path.join(   tagFolder, "TAG_%s.txt" % fileId)
    tagObj         = TagFile(tagFile)
    audio          = AudioSegment.from_file(wavFile, format="wav") # input audio
    totalLen       = len(audio)
    startTime      = 0
    timeSpecMs     = timeSpec * 1000
    while startTime + timeSpecMs < totalLen:
        print("    ----- %10.3lf" % (startTime / totalLen))
        segment = audio[startTime: startTime + timeSpecMs]
        outName = "VOCALSEG_%s_%s_%s.mp3" % (
            fileId, 
            TagFile.reparse(startTime // 1000), 
            TagFile.reparse((startTime + timeSpecMs) // 1000)
        )
        if tagObj.isInterlude(startTime // 1000, (startTime + timeSpecMs) // 1000):
            outFile = os.path.join(posMp3Folder, outName)
        else:
            outFile = os.path.join(negMp3Folder, outName)
        segment.export(outFile, format="mp3")
        startTime += timeSpecMs

if __name__ == "__main__":
    rootDir        = getRootFilePath()
    vocalWavFolder = os.path.join(rootDir, "VOCALWAV")
    for filename in getFileNameListInDir(vocalWavFolder):
        print("SplitAudioFile for filename = %s" % filename)
        SplitVocalAudioFile(filename, TIME_SPEC, rootDir)
        print("")
    print("Done.")