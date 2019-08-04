# Classifier_Performance_Evaluation_and_Comparison
Classifier performance evaluation and comparison based on mammographic mass data set

1. The mammographic mass data set (mammographic.csv) contains 961 instances of masses detected in mammograms, and contains the following
attributes:

 - BI-RADS assessment: 1 to 5 (ordinal)
 - Age: patient's age in years (integer)
 - Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
 - Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
 - Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
 - Severity: benign=No or malignant=Yes (binary)
 
BI-RADS is an assessment of how confident the severity classification is; it is not a "predictive"
attribute and so it will be discarded. The age, shape, margin, and density attributes are the features
that we will build our model with, and "severity" is the classification we will attempt to predict
based on those attributes.

Although "shape" and "margin" are nominal data types, which sklearn typically doesn't deal with
well, they are close enough to ordinal that we shouldn't just discard them. The "shape" for
example is ordered increasingly from round to irregular.

The data needs to be cleaned: many rows contain missing data. Some column needs to be
transformed to numerical data. Techniques such as KNN also require the input data to be
normalized first.

2. Purpose of this project: 
Applying the following supervised learning techniques, and see which
one yields the highest accuracy as measured with K-Fold cross validation (K=10).

  (1) Decision tree
  
        • Create a single train/test split of the data. Set aside 75% for training, and 25% for
          testing. Create a DecisionTreeClassifier and fit it to the training data. Measure the
          accuracy of the resulting decision tree model using the test data.
          
        • Use K-Fold cross validation to get a measure of your model’s accuracy (K=10). 
        
  (2) Random forest
  
        • Create a RandomForestClassifier using n_estimators=10 and use K-Fold cross validation
          to get a measure of the accuracy (K=10).

  (3) KNN
  
        • Create a neighbors.KNeighborsClassifier and use K-Fold cross validation to get a
          measure of the accuracy (K=10).
          
        • Try different values of K. Write a for loop to run KNN with K values ranging from 1 to
          50 and see if K makes a substantial difference. Make a note of the best performance you
          could get out of KNN.
          
  (4) Naive Bayes
  
        • Create a naïve_bayes.MultinomailNB and use K-Fold cross validation to get a measure of
          the accuracy (K=10).
  
  


