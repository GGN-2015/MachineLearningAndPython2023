# T-test workflow

Author: Guannan Guo(郭冠男)

To run this program, you need to make sure that `Python3` is installed on your computer, along with the pandas, numpy, and scipy modules. We will be using `ALL3.txt` as the test input for running this program, but since this file is large, it may not be included in the email.

## How to Run

We assume that you have installed all the necessary modules and placed the data file `ALL3.txt` in the same directory as `Code.py`. You can use the following command to start `Code.py` and obtain the top ten features ranked by `p-value`:

```bash
python Code.py ALL3.txt
```

If you want to use a different data file as the input for the program, you need to make sure that the data file meets the following conditions:
1. The first column of the data file contains the names of all the features.
2. The first row of the data file contains the class labels for all the samples ('POS' or 'NEG').
3. The adjacent columns of data should be separated by '\t' (tab) delimiter.
4. The remaining data in the file are floating-point numbers and do not contain NaN or invalid values.

## Structure of the Code

The function `fLoadDataMatrix` is used to load a data file and convert it into a matrix form, with the returned values being sample IDs, class labels, feature names, and the feature matrix.

The function `fSplitPosAndNeg` is used to split the feature matrix into positive and negative samples based on their class labels, with `posSet` being the feature matrix for the positive samples and `negSet` being the feature matrix for the negative samples.

The function `fT_test` is used to perform a two-sample T-test on the positive and negative feature matrices that have been split, resulting in t-values and p-values.

The function `fTopFeatureId` is used to return the IDs of the top N features with significant differences based on their p-values.

The function `fShowTopFeaturesInDataFile` is used to output the top N features with significant differences to the console.

In the main function, the script first obtains the data file name specified by the command line argument, and then calls the `fShowTopFeaturesInDataFile` function to perform feature extraction and output.
