#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class NaiveBayesClassifier:
    
    # initailizing X and Y here
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    
    # here we are calculating the prior probability according to the formula
    # label means class 
    def prior_prob(self,label):
        # here we are taking sum wherever Y has value equal to the label and then dividing it by total example
        return(np.sum(self.Y==label)/self.Y.shape[0])
    
    def conditional_prob(self,feature_column,feature_value,label):
        """here X is being filtered as sample for this conditional probability will the be those examples which 
        belongs to the given label."""
        X_filtered=self.X[self.Y==label]
        """here the favourable values will be those examples which have the given feature value in the given
           feature column given that they belongs to the above mentioned class."""
        return(np.sum(X_filtered[:,feature_column]==feature_value)/np.sum(self.Y==label))
    
    def posteriors(self,X_test):
        # calculating total number of classes
        classes=np.unique(self.Y)  
        # total numbers of columns gives us the total number of features , so we are calculating it below.
        no_of_features=self.X.shape[1]
        # creating an empty list to save the values of posterior probabilities.
        post=[]               
        for label in classes:
            likelihood=1.0
            for feature_column in range(no_of_features):
                # here we are calculating the value of the likelihood as per the formula mentioned in read me. 
                likelihood*=self.conditional_prob(feature_column,X_test[feature_column],label)
            # here we are appending a tuple containing the label and likelihood associated with that label.    
            post.append((label,likelihood*self.prior_prob(label)))
            
        """as the tuple have label and likelihood and we need to take max of likelihood, so we find that max value 
        using the below syntax and then we find its index."""
        max_ind=post.index(max(post,key=lambda tup:tup[1]))
        
        #returning the class here which is indexed at zero for each tuple(class) in the list.
        return(int(post[max_ind][0]))
    
    def predict(self,X_test):
        #creating an empty list to save the labels/classes for test cases.
        pred=[]   
        for i in range(X_test.shape[0]):
            val=self.posteriors(X_test[i])
            pred.append(val)
        return(pred)

    #creating an accuracy function to check how well our classifier is working.
    def accuracy(self,X_test,Y_test):
        pred=self.predict(X_test)
        #accuracy is scaled between 0 to 1.
        return(np.mean(pred==Y_test))
  
    
#Example
#using pandas we read the dataset
df=pd.read_csv("mushrooms.csv")
labelencoder=LabelEncoder()
#here labelencoder is used to convert the categorical data into numerical form
ds=df.apply(labelencoder.fit_transform)
#Creating X and Y
X=ds.values[:,1:]
Y=ds.values[:,0]
#using train_test_split we here splitted our data into training and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
#creating an instance here
NBC=NaiveBayesClassifier(X_train,Y_train)
acc=NBC.accuracy(X_test,Y_test)   #to check the accuracy of our classifier
# here accuracy for this dataset is about 99%.
print(acc)


# In[ ]:




