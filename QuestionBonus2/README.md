# README

This project is a Python script that uses Support Vector Machine (SVM) to classify data. It requires the following libraries to be installed: pandas, numpy, scipy, and matplotlib.

## Data

The input data is a tab-separated file containing samples with features and classes. The file should have a header row with the feature names and a row for each sample. The first column of each row should contain the sample name, and the last column should contain the class label ("POS" or "NEG"). All other columns should contain feature values.

## Functions

The script contains several functions for data processing, feature selection, SVM training, and performance evaluation. Each function is briefly described below:

- `fLoadDataMatrix(filename: str)` loads the data file in a numpy matrix format and returns the sample names, class labels, feature names, and feature values.
- `fSplitPosAndNeg(tMatrix, tClass)` separates the positive and negative samples from the input matrix based on their class labels.
- `fT_test(posSet, negSet)` performs a t-test on the positive and negative samples to determine which features have significantly different means between the two classes.
- `fTopFeatureId(pValue, topN = 10)` returns the top N features with the smallest p-values from the t-test.
- `fGetTopFeaturesInDataFile(filename: str) -> list` returns the top features for the input data file.
- `fRunSvm(xTrain: np.ndarray, yTrain: np.ndarray, xTest: np.ndarray, yTest: np.ndarray)` trains an SVM model on the input training data and returns the confusion matrix (true negatives, false positives, false negatives, true positives) for the test data.
- `runSvmOnFeatureSet(featureSet, tMatrix, tClass)` trains an SVM model on a subset of features and returns the mean and standard deviation of the F1 score and accuracy over multiple runs.
- `fGetMeanStdList(topFeatureId, tMatrix, tClass, maxFeatureCnt = MAX_FEATURE_CNT)` returns the mean and standard deviation of the F1 score and accuracy for subsets of the top features with increasing sizes.
- `fPlotLine(meanAccList, stdAccList, meanF1List, stdF1List)` plots two lines representing the mean and standard deviation of the accuracy and F1 score for subsets of the top features with increasing sizes.

## Usage

To use the script, simply run it in a Python environment with the required libraries installed. The `filename` variable in the main block should be set to the path of the input data file. The script will output a plot showing the mean and standard deviation of the accuracy and F1 score for subsets of the top features with increasing sizes. The number of features to consider can be adjusted by changing the `MAX_FEATURE_CNT` variable.
