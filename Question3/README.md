# Question3 fLoadDataMatrix

```
Author   : Guannan Guo(郭冠男)
StudentId: 21200612
```

## Run and Environment

In this project, we have implemented a Python3 program that reads CSV 
files. This functionality will be implemented using panda's CSV reading 
function in the next project. To run this program, you need to make 
sure that Python3 is correctly installed on your computer and that the 
numpy module is installed. To run this program, you need to enter the 
following command in the command line:

```bash
python3 Code.py filename.csv
```

Where `filename.csv` should be a valid CSV file. We require a valid CSV 
file to satisfy the following three conditions:

The file should have at least one line of content;
Each line of the file should have the same number of fields;
All fields should be separated by commas, and please note that our 
program does not support escape characters.
In the structure of a CSV file, we refer to the first line of the file 
as the header. The header should have a field called "class" 
(case-insensitive), which is referred to as the class label. The first 
field in the header is usually empty but can also be non-empty, and it 
is considered as the name identifier of the sample. All other fields 
are considered as data fields, which means that the corresponding 
column features should be recorded in a numerical way. When the file 
inputted to the program does not meet our requirements, the program 
will use AssertionError to report an error. The user of the 
fLoadDataMatrix function can use `try` to obtain the corresponding error 
information.

## About TestData

Our project provides two valid CSV files and one invalid CSV 
file. `TestData1.csv` is a valid CSV file that contains data from the 
example in a PPT. `TestData2.csv` is also a valid CSV file that contains 
partial grades of some students in the Computer Science Department in 
the second semester of their sophomore year. `TestData3.csv` is an 
invalid CSV file because some of its rows have a different number of 
fields compared to other rows. When the program reads a valid CSV file, 
it outputs four numpy arrays, `Sample`, `Class`, `Feature`, and `Matrix`, 
to the screen for testing. `Sample`, `Class`, and `Feature` are 
one-dimensional vectors, and Matrix is a two-dimensional data matrix 
where each column contains all features of a sample.

Please refer to the demonstration screenshots `PrintScreen1.png`, 
`PrintScreen2.png`, and `PrintScreen3.png` for the read-in results of 
the three files.
