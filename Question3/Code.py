import sys
import numpy

def fFindColumnId(listOfString, columnName):
    assert type(columnName) == str
    assert type(listOfString) == list
    for item in listOfString:
        assert type(item) == str
    columnId = -1 # -1 means not found
    for i in range(len(listOfString)):
        item = listOfString[i]
        if item.lower() == columnName.lower():
            columnId = i
    return columnId

# usage:
# (Sample, Class, Feature, Matrix) = fLoadDataMatrix(FileName)
def fLoadDataMatrix(filename):
    if type(filename) != str:
        raise AssertionError("fLoadDataMatrix: filename should be a string.")
    try:
        fp = open(filename, encoding="utf-8")
        openFileSuc = True
    except:
        openFileSuc = False
    if not openFileSuc:
        raise AssertionError("fLoadDataMatrix: fail to open file <%s>" % filename)
    lines = []
    for line in fp: # read in every line, and skip the empty lines
        if line.strip() == "":
            continue
        lineNow = []
        for item in line.split(','): # CSV
            lineNow.append(item.strip())
        lines.append(lineNow)
    # file assertions
    if len(lines) <= 0:
        raise AssertionError("fLoadDataMatrix: csv file should have at least one line.")
    columnIdClass = fFindColumnId(lines[0], "class")
    if columnIdClass == -1:
        raise AssertionError("fLoadDataMatrix: `class` column not found.")
    if columnIdClass == 0:
        raise AssertionError("fLoadDataMatrix: `class` column shouldn't be the first column")
    for i in range(1, len(lines)):
        if len(lines[i]) != len(lines[0]):
            raise AssertionError("fLoadDataMatrix: line %d length error." % (i + 1))
    # we assume that the first column is the object name
    tSample  = []
    tClass   = []
    tFeature = []
    tMatrix  = []
    for lineId in range(1, len(lines)): # get sample names
        tSample.append(lines[lineId][0])
    for lineId in range(1, len(lines)): # get class names
        tClass.append(lines[lineId][columnIdClass])
    for columnId in range(1, len(lines[0])): # get features name
        if columnId == columnIdClass:
            continue
        tFeature.append(lines[0][columnId])
    for columnId in range(1, len(lines[0])):
        if columnId == columnIdClass: # jump the class column
            continue
        columnNow = []
        for lineId in range(1, len(lines)):
            columnNow.append(lines[lineId][columnId])
        tMatrix.append(columnNow)
    return (
        numpy.array(tSample ),
        numpy.array(tClass  ),
        numpy.array(tFeature),
        numpy.array(tMatrix , dtype=numpy.float64),
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise AssertionError("Usage: python3 Code.py filename.csv")
    (Sample, Class, Feature, Matrix) = fLoadDataMatrix(sys.argv[1])
    print("Sample :   ",  Sample)
    print("Class  :   ",   Class)
    print("Feature:   ", Feature, end="\n\n")
    print("Matrix : \n",  Matrix)
