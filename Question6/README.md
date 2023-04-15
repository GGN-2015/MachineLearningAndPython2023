# README

## Prerequisites

This code imports various machine learning models and libraries from Scikit-Learn and other Python libraries. 

The imported machine learning models are:

- Support Vector Machines (SVM)
- LassoCV
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier

The imported libraries are:

- NumPy
- SciPy
- Matplotlib
- Pandas
- os

These libraries are used for data manipulation, statistical analysis, and visualization.

## Main Function

There is a function called `fMainPlotFunction`, which takes a filename as input and produces a plot with four subplots. The subplots show the performance of the models mentioned above on the top 1, top 10, top 100, and end 100 features.

To use this function, you need to have a data matrix file in a specified format. The function `fLoadDataMatrix` is used to load the data matrix from the file. The function `fGetTopFeatures` is used to get the indices of the top features based on a T-test. In our program, we use `ALL3.txt` as our input. In the program, we use $30\%$ of the data as the training set and $70\%$ of the data as the test set. We repeat the experiments ten times for each model and calculate the average performance metrics to finally plot a histogram.

To call the `fMainPlotFunction` function, you can use the following code:

```python
filename = 'data_matrix.txt'
fMainPlotFunction(filename)
```

This will create a plot with four subplots showing the model performance on the top 1, top 10, top 100, and end 100 features based on the data matrix in the `data_matrix.txt` file, where adjacent data in the file is separated by `tab`.

