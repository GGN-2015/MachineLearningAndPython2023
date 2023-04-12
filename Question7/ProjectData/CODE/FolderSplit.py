import os
from SampleRateChecker import getRootFilePath
from shutil import copyfile

# 获取一个目录下的文件名列表
# fileNameList = getFileNameListInDir(dirname)
def getDirNameListInDir(dirname):
    ans = []
    for filename in os.listdir(dirname):
        if os.path.isdir(os.path.join(dirname, filename)):
            ans.append(filename)
    return ans

def splitTmpMp3Folder():
    tmpMp3Dir   = os.path.join(rootDir, "TMPMP3")
    vocalMp3Dir = os.path.join(rootDir, "VOCALMP3")
    bgmMp3Dir   = os.path.join(rootDir, "BGMMP3")
    for filename in getDirNameListInDir(tmpMp3Dir):
        print("copying dirname = %s\n" % filename)
        tmpVocalFile = os.path.join(tmpMp3Dir, filename + "/vocals.wav")
        tmpBgmFile   = os.path.join(tmpMp3Dir, filename + "/accompaniment.wav")
        copyfile(tmpVocalFile, os.path.join(vocalMp3Dir, filename + ".wav"))
        copyfile(  tmpBgmFile, os.path.join(  bgmMp3Dir, filename + ".wav"))

if __name__ == "__main__":
    rootDir   = getRootFilePath()
    splitTmpMp3Folder()