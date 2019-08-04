# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:56:13 2019

@author: Yaya Liu

"""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt 

import statistics
    
from sklearn import model_selection
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix  
#from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB



def data_prepare():
    ColumnNames = ['Age', 'Shape', 'Margin', 'Density', 'Severity']  # Import data
    
    mammogram = pd.read_table('./mammographic.csv', sep = ',', usecols = [1, 2, 3, 4, 5], 
                              engine = 'python', header = None, names = ColumnNames)
    
    mammogram.replace(['?'], np.nan, inplace = True)  
    
    print("Number of missing values in each column: ")
    print(mammogram.isnull().sum())  # show missing values in each column
    
    mammogram = mammogram.dropna()      # Drop missing values 
    
    #print(mammogram.dtypes)
    mammogram.Age = pd.to_numeric(mammogram.Age, errors = 'coerce')  # change data type from "object" to "int"
    mammogram.Shape = pd.to_numeric(mammogram.Shape, errors = 'coerce') 
    mammogram.Margin = pd.to_numeric(mammogram.Margin, errors = 'coerce')
    mammogram.Density = pd.to_numeric(mammogram.Density, errors = 'coerce')    
    mammogram['Severity'] = mammogram['Severity'].map({'yes': 1, 'no': 0}) # Transfer Severity "yes" to 1, and "no" to 0
    
    #print(mammogram.dtypes)
    #print(mammogram.head)
    print("Data set dimensions : {}".format(mammogram.shape))

    return mammogram 

def data_visualization(mammogram):
    sns.set(style = 'whitegrid', context = 'notebook', color_codes=True)
    col = ['Age', 'Shape', 'Margin', 'Density']
    
    # Pair plot shows the relationship between two variables
    sns.pairplot(mammogram[col], height = 2.5)
    g = sns.pairplot(mammogram[col], height = 2.5)
    g.fig.suptitle("Paired plot between attributes", y = 1)
    plt.show()
    
    # Bar plot shows the number of benign and malignant cases based on column "Severity"
    sns.countplot(x = 'Severity',data = mammogram)
    plt.show()
    
    # Correlation matrix heat map
    corr = mammogram.corr()
    
    sns.set(font_scale = 1.15)
    plt.figure(figsize = (14, 10))
    
    sns.heatmap(corr, vmax = .8, linewidths = 0.01, square = True, 
                annot = True, cmap = 'YlGnBu',linecolor = "black")
    plt.title('Correlation between attributes');
    plt.show()
    

# Split data into training set and test set
def data_split(mammogram):

    X = mammogram.values[:, 0:3] 
    Y = mammogram.values[:, 4]               # Seperating the target variable 
  
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = None) 
    return X, Y, X_train, X_test, y_train, y_test
    
# Decision Treee
def my_DecisionTree(X, Y, X_train, X_test, y_train, y_test):
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", max_depth = 3, min_samples_leaf = 5)
    
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    
    # Predicton on test with giniIndex 
    y_pred = clf_gini.predict(X_test) 
    print("Decision Tree (Single Split) / Accuracy: %.3f%%" %(accuracy_score(y_test,y_pred)*100))  # print accuracy    
    #print("Decision Tree/Confusion Matrix:")  # print confusion matrix
    #print(confusion_matrix(y_test, y_pred)) 
    #print(classification_report(y_test, y_pred))  
    
    # K-fold cross validation
    kfold = model_selection.KFold(n_splits = 10, random_state = None, shuffle = False)
    results = model_selection.cross_val_score(clf_gini, X, Y, cv = kfold)
    print("Decision Tree (10 K-Fold) / Mean Accuracy: %.3f%%" % (results.mean() * 100))
   
# Random Forest
def my_RandomForest(X, Y):
    kfold = model_selection.KFold(n_splits = 10, random_state = None, shuffle = False)
    clf = RandomForestClassifier(n_estimators = 10, random_state = None, max_depth = 3, min_samples_leaf = 5)
    results = model_selection.cross_val_score(clf, X, Y, cv = kfold)
    print("Random Forest (10 K-Fold) / Mean Accuracy: %.3f%%" % (results.mean() * 100))    

# Multinomial Naive Bayes classifier  
def my_MultinomialNB(X, Y):    
    mnb = MultinomialNB(alpha = 0.2)
#    mnb.fit(X_train, y_train)
#    y_pred = mnb.predict(X_test)
#    print("MultinomialNB (Single Split) / Accuracy: %.3f%%" %(accuracy_score(y_test, y_pred)*100))
    
    kfold = model_selection.KFold(n_splits = 10, random_state = None, shuffle = False)    
    results = model_selection.cross_val_score(mnb, X, Y, cv = kfold)
    print("MultinomialNB (10 K-Fold) / Mean Accuracy: %.3f%%" % (results.mean() * 100)) 

# K nearest neighbors    
def my_KNN(mammogram):
    X = mammogram.values[:, 0:3]
    sta = StandardScaler()  # normalize data
    input = sta.fit_transform(X.astype(float))
    #print(input)
    
    Y = mammogram['Severity'].values
    
    knn = KNeighborsClassifier(n_neighbors = 10)
    kfold = model_selection.KFold(n_splits = 10, random_state = None, shuffle = False)    
    results = model_selection.cross_val_score(knn, input, Y, cv = kfold)
    print("10 Nearest Neighbors (10 K-Fold) / Mean Accuracy: %.3f%%" % (results.mean() * 100))

    # Calculate accuracy mean when k from 1 to 50, and store it in dictionary "resultsDict"  
    resultsDict = {}
    for k in range(1, 51):
           knn = KNeighborsClassifier(n_neighbors = k)
           kfold = model_selection.KFold(n_splits = 10, random_state = None, shuffle = False)
           results = model_selection.cross_val_score(knn, input, Y, cv = kfold)
           resultsDict[k] = results.mean()
     
    #print("N Nearest Neighbors (10 K-Fold)/{N, Accuracy}; ")
    #print(resultsDict)    

    numbers = [resultsDict[key] for key in resultsDict]
    
    bestAccuracy = max(numbers)   # Find the highest accuracy
    
    # Find which k gave the best accuracy
    for k, accuracy in resultsDict.items():
        if accuracy == bestAccuracy:
            bestK = k
            
    print("When k = %d, we can get the hightest accuracy." %bestK, 
          "The highest accuracy is %.3f%%" %(bestAccuracy * 100))             
      
    k_mean = round(statistics.mean(numbers), 5)  
    print("Mean of the accuracy from k = 1 to 50 is %.3f%%" %(k_mean * 100))
    
    k_sd = round(statistics.pstdev(numbers),3)
    print("Standard deviation of the accuracy from k = 1 to 50: ", k_sd)
    
    # plot scatterplot which shows the relationship between k value and accuracy
    x_axis = [x for x in range(1, 51)]
    plt.scatter(x_axis, numbers)
    plt.title("Relationship between K Value and Accuray")
    plt.xlabel("Number of Neighbors K")
    plt.ylabel("Accuracy")
    plt.show()
         
       
if __name__ == '__main__':
    mammogram = data_prepare()
    data_visualization(mammogram)
    
    X, Y, X_train, X_test, y_train, y_test = data_split(mammogram)
    
    my_DecisionTree(X, Y, X_train, X_test, y_train, y_test)
    my_RandomForest(X, Y)    
    my_KNN(mammogram)
    my_MultinomialNB(X, Y)
    
    
    
    


