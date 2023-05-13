## README

This is a Python script that performs a feature selection and classification analysis on a given dataset. The dataset is assumed to be in tab-separated format, with the first row containing feature names and the first column containing sample names.

## Dependencies

- pandas
- numpy
- scipy
- matplotlib
- scikit-learn

The script contains several functions that can be called to perform various operations on the dataset:

1. **fLoadDataMatrix**(filename: str) -> tuple:
   This function takes a filename as input and returns a tuple containing four elements:
   - tSample: an array of sample names
   - tClass: an array of class labels
   - tFeature: an array of feature names
   - tMatrix: a matrix of data, where each row represents a sample and each column represents a feature.
2. **fSplitPosAndNeg**(tMatrix, tClass) -> tuple:
   This function takes a matrix of data and an array of class labels as input and splits the data into two sets based on the class labels: a positive set and a negative set.
3. **fT_test**(posSet, negSet) -> tuple:
   This function takes a positive set and a negative set as input and performs a two-sample t-test to identify features that are significantly different between the two sets. It returns two arrays: one containing the t-values and one containing the p-values for each feature.
4. **fTopFeatureId**(pValue, topN = 10) -> list:
   This function takes an array of p-values as input and returns a list of the top N features with the lowest p-values.
5. **fGetTopFeaturesInDataFile**(filename: str) -> list:
   This function takes a filename as input, loads the data using fLoadDataMatrix, splits the data into positive and negative sets using fSplitPosAndNeg, performs a t-test using fT_test, and returns a list of the top features using fTopFeatureId.
6. **fRunSvm**(xTrain: np.ndarray, yTrain: np.ndarray, xTest: np.ndarray, yTest: np.ndarray) -> tuple:
   This function takes training and testing data and labels as input and trains an SVM classifier using scikit-learn's svm.SVC class with a radial basis function kernel. It returns a tuple containing the true negatives, false positives, false negatives, and true positives for the classifier.
7. **fRunNbayes**(xTrain: np.ndarray, yTrain: np.ndarray, xTest: np.ndarray, yTest: np.ndarray) -> tuple:
   This function takes training and testing data and labels as input and trains a Naive Bayes classifier using scikit-learn's GaussianNB class. It returns a tuple containing the true negatives, false positives, false negatives, and true positives for the classifier.
8. **fRunKNN**(xTrain: np.ndarray, yTrain: np.ndarray, xTest: np.ndarray, yTest: np.ndarray, k = 5) -> tuple:
   This function takes training and testing data and labels as input, as well as a value for k for the KNN classifier, and trains the classifier using scikit-learn's KNeighborsClassifier class. It returns a tuple containing the true negatives, false positives, false negatives, and true positives for the classifier.
