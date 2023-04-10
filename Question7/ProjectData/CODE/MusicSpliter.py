from pydub import AudioSegment
import os

from SampleRateChecker import getRootFilePath, getFileNameListInDir
TIME_SPEC = 2 # second

class TagFile:
    def getTime(tm):
        m, s = list(map(int, tm.split(":")))
        return m * 60 + s
    def parse(line):
        f, t = line.split()
        return (TagFile.getTime(f), TagFile.getTime(t))
    def __init__(self, filename):
        self.timeList = []
        for line in open(filename):
            if line.strip() != "":
                self.timeList.append(TagFile.parse(line))
    def debugOutput(self):
        for timeSpec in self.timeList:
            print(timeSpec)
    def isInterlude(self, tFr: int, tTo: int):
        for f, t in self.timeList:
            if f <= tFr and tFr <= tTo and tTo <= t:
                return True
        return False
    def reparse(tm: int):
        return "%02d-%02d" % (tm // 60, tm % 60)

# 将音频分割为以 timeSpecMs 为单位长度的片段
def SplitAudioFile(filename, timeSpec: int, rootFolder: str):
    fileId       = filename.split('.')[0]
    modMp3Folder = os.path.join(rootFolder, "MODMP3")
    posMp3Folder = os.path.join(rootFolder, "SEG/POS")
    negMp3Folder = os.path.join(rootFolder, "SEG/NEG")
    tagFolder    = os.path.join(rootFolder, "TAG")
    mp3File      = os.path.join(modMp3Folder, filename)
    tagFile      = os.path.join(   tagFolder, "TAG_%s.txt" % fileId)
    tagObj       = TagFile(tagFile)
    audio        = AudioSegment.from_file(mp3File, format="mp3") # input audio
    totalLen     = len(audio)
    startTime    = 0
    timeSpecMs   = timeSpec * 1000
    while startTime + timeSpecMs < totalLen:
        print("    ----- %10.3lf" % (startTime / totalLen))
        segment = audio[startTime: startTime + timeSpecMs]
        outName = "SEG_%s_%s_%s.mp3" % (
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
    rootDir      = getRootFilePath()
    modMp3Folder = os.path.join(rootDir, "MODMP3")
    for filename in getFileNameListInDir(modMp3Folder):
        print("SplitAudioFile for filename = %s" % filename)
        SplitAudioFile(filename, TIME_SPEC, rootDir)
        print("")
    print("Done.")