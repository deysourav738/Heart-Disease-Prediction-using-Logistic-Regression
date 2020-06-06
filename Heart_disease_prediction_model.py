#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction Model Using Logistic Regression
# 
#    Author: Sourav Dey

# In[153]:


#importing the necessary modules

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle


# In[154]:


# Reading the dataset

data=pd.read_csv(r"/Users/souravdey/Desktop/heart.csv")


# In[155]:


# Checking for null values

data.isnull().sum()


# In[156]:


data.describe()


# In[157]:


data


# In[158]:


# To find the categorical features

cat_feat=[]
for i in data.columns:
    if len(data.groupby(i)) <= 5:
        cat_feat.append(i)


# In[159]:


cat_feat


# In[160]:


# Histograms
data['age'].hist(edgecolor='black',linewidth=2);


# In[161]:


# Number of rows and columns in the dataset 

data.shape


# In[162]:


fig=plt.figure(figsize=(20,20))
sns.heatmap(data.corr(),annot=True)


# In[163]:


data['target'].hist()


# In[164]:


data.columns


# In[165]:


data


# In[166]:


# Boxplots for checking outliers in the data

fig = plt.subplots(figsize=(16,8))
sns.boxplot(data=data)


# In[167]:


# Removing the outliers from the dataset

data=data.drop(data[(np.abs(stats.zscore(data))>3)].index)
data


# In[168]:


# Checks proportion of data belongs to each class
plt.pie(x=[len(data[data['target']==0]),len(data[data['target']==1])],labels=['Not Diseased','Diseased'],autopct='%1.2f%%',startangle=90)
plt.show()


# In[169]:


# X contains all the independent variables
# Y contains the dependent variable

x=pd.DataFrame(data.iloc[:,0:13])
y=pd.DataFrame(data.iloc[:,13])
x,y


# In[170]:


# Create model for checking feature importances
model = ExtraTreesClassifier(n_estimators=100)

# Fits the data into the model
model.fit(x,y)

print(model.feature_importances_)


# In[171]:


# Sort feature importances in descending order
indices = np.argsort(model.feature_importances_)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [x.columns[i] for i in indices]

# Create plot
plt.figure()

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(x.shape[1]), model.feature_importances_[indices])

# Add feature names as x-axis labels
plt.xticks(range(x.shape[1]), names, rotation=90)

# Show plot
plt.show()


# In[172]:


# Droping some less important features

x=x.drop(labels=names[11:],axis=1)
x


# In[173]:


# scaling object
sc=StandardScaler()

# scaling the data by fit_transform method
x=sc.fit_transform(x)


# In[174]:


# Splits the data into train and test in 8.5 : 1.5 ratio 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.15,random_state=42)


# ### Logistic Regression

# In[175]:


# Logistic regression object

test_reg=LogisticRegression(class_weight='balanced',random_state=42)


# In[176]:


# Hperparameter tunning

param_grid = { 
    'C': [0.1,0.2,0.3,0.4],
    'penalty': ['l1', 'l2'],
    'class_weight':[{0: 1, 1: 1},{ 0:0.67, 1:0.33 },{ 0:0.75, 1:0.25 },{ 0:0.8, 1:0.2 }]}
CV_rfc = GridSearchCV(estimator=test_reg, param_grid=param_grid, cv= 5)
CV_rfc.fit(x_train, y_train.values.ravel())
CV_rfc.best_params_


# In[177]:


# passing the best parameters

reg = LogisticRegression(C=0.4,random_state=42,penalty='l2',class_weight={0: 1, 1: 1})
print("Accuracy on training set:")
res=cross_val_score(reg,x_train,y_train.values.ravel(),cv=10,scoring="accuracy")
print("Average accuracy:\t{0:}\n".format(np.mean(res)))
print("Standard Deviation:\t{0:}\n".format(np.std(res)))


# In[178]:


reg.fit(x_train,y_train.values.ravel())
y_pred = reg.predict(x_test)


# In[179]:


plt.title("Accuracy Distribution of The Model in 10 Fold Crossvalidation")
plt.hist(res,edgecolor='black');


# In[180]:


print("Accuracy:\t{0:}\n".format(accuracy_score(y_test,y_pred)))


# In[181]:


tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d",cmap='Accent')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix for Logistic Regression')
plt.show()
print("True Negative:\t{}\t False Positive:\t{}\t False Negative:\t{}\t True Positive:\t{}\t".format(tn,fp,fn,tp))


# In[182]:


print("Classification Report:")
print(classification_report(y_test,y_pred))


# In[183]:


# save the the model and the scaling object

pickle.dump(reg,open('heart_disease_prediction_model.pkl','wb'))
pickle.dump(sc,open('heart_disease_prediction_scaler.pkl','wb'))


# In[ ]:




