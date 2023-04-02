# Feature Selection using T-Test

This Python program performs feature selection using T-test based on two groups of samples with binary class labels. It reads a data file in tab-delimited format and extracts the matrix of data samples, the binary class labels, and the feature names. It then splits the data matrix into two sets based on the class labels and performs T-test on each feature to calculate the p-values. The program outputs a ranked list of features based on their p-values and also plots scatter plots of selected feature pairs.

## Prerequisites

This program requires the following Python libraries:
- pandas
- numpy
- scipy
- matplotlib

## Usage

To run the program, use the following command:

```
python feature_selection.py filename
```

where `filename` is the name of the tab-delimited data file. We used `ALL3.txt` as the sample input to generate `PrintScreen.png`.

## Output

The program calculates the list of top-ranked features based on their p-values and plots scatter plots of selected feature pairs. The scatter plots show the distribution of the two groups of samples with different colors and labels.
