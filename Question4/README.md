# T-test workflow

Author: Guannan Guo(郭冠男)

To run this program, you need to make sure that `Python3` is installed on your computer, along with the pandas, numpy, and scipy modules. We will be using `ALL3.txt` as the test input for running this program, but since this file is large, it may not be included in the project folder.

## How to Run

We assume that you have installed all the necessary modules and placed the data file `ALL3.txt` in the same directory as `Code.py`. You can use the following command to start `Code.py` and obtain the top ten features ranked by `p-value`:

```bash
python Code.py ALL3.txt
```

If you want to use a different data file as the input for the program, you need to make sure that the data file meets the following conditions:
1. The first column of the data file contains the names of all the features.
2. The first row of the data file contains the class labels for all the samples ('POS' or 'NEG').
3. The adjacent columns of data are separated by '\t' (tab) delimiter.
4. The remaining data in the file are floating-point numbers and do not contain NaN or invalid values.

## Structure of this Project

In the `Code.py` file, there are four functions:
1. fLoadDataMatrix: used to obtain data from a file
2. fSplitPosAndNeg: used to separate positive and negative samples in the data matrix
3. fT_test: used to calculate t-value and p-value based on the separated positive and negative samples
4. fTopFeatureId: calls the above three functions to complete the entire process of t-test.
