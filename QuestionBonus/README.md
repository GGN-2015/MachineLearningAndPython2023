# README

This code performs an analysis of the accuracy of three classifiers (`SVM, Naive Bayes`, and `KNN`) using different numbers of top features. The data used in this analysis is loaded from the file "`ALL3.txt`". 

To run the code, make sure that the necessary dependencies (e.g., `numpy, matplotlib, seaborn`) are installed. Then, simply execute the script in a Python environment. 

The script first calls the function "`fGetTopFeatures`" to retrieve the top features from the data file. It then loads the data matrix from the file using the function "`fLoadDataMatrix`". The loaded data is transposed to make it easier to work with. 

Next, the script creates three empty lists to store the accuracy values for each classifier. It then loops through a range of feature counts (from 1 to a maximum feature count defined by the variable "`MAX_FEATURE_CNT=100`"), and for each feature count, it selects the top features and calculates the accuracy of each classifier using those features. The accuracy values are appended to the corresponding list.

Finally, the script creates a heatmap using the Seaborn library to display the accuracy values for each classifier and each feature count. The heatmap shows the accuracy values using a color scale (`cmap="YlOrRd"`), with darker colors indicating higher accuracy. The heatmap is labeled with the type of classifier on the y-axis and the number of features used on the x-axis.

## Analysis

From the results of the program, it is easy to see that when the top three features are selected for training, a relatively good training effect can be obtained. As the number of features increases, the performance of the `Naive Bayes` algorithm is the best in the three classifiers.

