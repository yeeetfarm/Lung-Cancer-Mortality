#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('lung_cancer_mortality_data_small.csv')
df


# In[2]:


df[['age','bmi','cholesterol_level']].describe()


# In[3]:


df.groupby('cancer_stage')['survived'].sum().plot(kind='bar',xlabel='Cancer Stage',ylabel='Number of Survivors',title='Survivors by Cancer Stage')


# In[4]:


df.groupby('smoking_status')['survived'].sum().plot(kind='bar',xlabel='Smoking Status',ylabel='Number of Survivors',title='Survivors by Smoking Status')


# In[5]:


df.groupby('family_history')['survived'].sum().plot(kind='bar',xlabel='Family History',ylabel='Number of Survivors',title='Survivors by Family History')


# In[6]:


df.groupby('gender')['survived'].sum().plot(kind='bar',xlabel='Gender',ylabel='Number of Survivors',title='Survivors by Gender')


# In[7]:


#Descending Survival Rate by Treatment Type
SRbytreatment=df.groupby('treatment_type')['survived'].agg(total='count',survivors=lambda x:(x==1).sum())
SRbytreatment['survival rate']=SRbytreatment['survivors']/SRbytreatment['total']
SRbytreatment.sort_values('survival rate',ascending=False)


# In[8]:


#Descending Survival Rate by Country
SRbycountry=df.groupby('country')['survived'].agg(total='count',survivors=lambda x:(x==1).sum())
SRbycountry['survival rate']=SRbycountry['survivors']/SRbycountry['total']
SRbycountry.sort_values('survival rate',ascending=False)


# In[9]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report
labelencoder=LabelEncoder()
df['gender encoded']=labelencoder.fit_transform(df['gender'])
df['cancer stage encoded']=labelencoder.fit_transform(df['cancer_stage'])
df['family history encoded']=labelencoder.fit_transform(df['family_history'])
df['smoking status encoded']=labelencoder.fit_transform(df['smoking_status'])
df['treatment type encoded']=labelencoder.fit_transform(df['treatment_type'])
#x=df[['age','bmi','cholesterol_level','hypertension','asthma','cirrhosis','other_cancer']]
x=df[['age','gender encoded','cancer stage encoded','family history encoded','smoking status encoded','bmi','cholesterol_level','hypertension','asthma','cirrhosis','other_cancer']]
y=df['survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=42)
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
#accuracy=overall correctness
accuracy=accuracy_score(y_test,y_pred)
#precision=ability to correctly predict positive class
precision=precision_score(y_test,y_pred)
#recall=quantifies ability to correctly predict positive class
recall=recall_score(y_test,y_pred)
#confmatrix=[TP FP][FN TN]
confmatrix=confusion_matrix(y_test,y_pred)
classreport=classification_report(y_test,y_pred)
print('Accuracy: ',accuracy)
print('Precision: ',precision)
print('Recall: ',recall)
print('Confusion Matrix: ',confmatrix)
print('Classification Report: ',classreport)


# In[10]:


#Random Forest Classifier: builds trees independently using random subset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,r2_score
labelencoder=LabelEncoder()
df['gender encoded']=labelencoder.fit_transform(df['gender'])
df['cancer stage encoded']=labelencoder.fit_transform(df['cancer_stage'])
df['family history encoded']=labelencoder.fit_transform(df['family_history'])
df['smoking status encoded']=labelencoder.fit_transform(df['smoking_status'])
df['treatment type encoded']=labelencoder.fit_transform(df['treatment_type'])
#x=df[['age','bmi','cholesterol_level','hypertension','asthma','cirrhosis','other_cancer']]
x=df[['age','gender encoded','cancer stage encoded','family history encoded','smoking status encoded','bmi','cholesterol_level','hypertension','asthma','cirrhosis','other_cancer']]
y=df['survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=42)
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
#accuracy=overall correctness
accuracy=accuracy_score(y_test,y_pred)
#precision=ability to correctly predict positive class
precision=precision_score(y_test,y_pred)
#recall=quantifies ability to correctly predict positive class
recall=recall_score(y_test,y_pred)
#confmatrix=[TP FP][FN TN]
confmatrix=confusion_matrix(y_test,y_pred)
classreport=classification_report(y_test,y_pred)
print('Accuracy: ',accuracy)
print('Precision: ',precision)
print('Recall: ',recall)
print('Confusion Matrix: ',confmatrix)
print('Classification Report: ',classreport)


# In[11]:


#Random Forest Factors' Importance
rfimportance=pd.DataFrame()
rfimportance['variables']=x.columns
rfimportance['importance']=rf.feature_importances_
rfimportance.sort_values(by='importance',ascending=False)


# In[12]:


#Gradient Boost: builds trees sequentially, correcting errors of previous one
from xgboost import XGBClassifier
labelencoder=LabelEncoder()
df['gender encoded']=labelencoder.fit_transform(df['gender'])
df['cancer stage encoded']=labelencoder.fit_transform(df['cancer_stage'])
df['family history encoded']=labelencoder.fit_transform(df['family_history'])
df['smoking status encoded']=labelencoder.fit_transform(df['smoking_status'])
df['treatment type encoded']=labelencoder.fit_transform(df['treatment_type'])
#x=df[['age','bmi','cholesterol_level','hypertension','asthma','cirrhosis','other_cancer']]
x=df[['age','gender encoded','cancer stage encoded','family history encoded','smoking status encoded','bmi','cholesterol_level','hypertension','asthma','cirrhosis','other_cancer']]
y=df['survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=42)
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
y_pred=xgb.predict(x_test)
#accuracy=overall correctness
accuracy=accuracy_score(y_test,y_pred)
#precision=ability to correctly predict positive class
precision=precision_score(y_test,y_pred)
#recall=quantifies ability to correctly predict positive class
recall=recall_score(y_test,y_pred)
#confmatrix=[TP FP][FN TN]
confmatrix=confusion_matrix(y_test,y_pred)
classreport=classification_report(y_test,y_pred)
print('Accuracy: ',accuracy)
print('Precision: ',precision)
print('Recall: ',recall)
print('Confusion Matrix: ',confmatrix)
print('Classification Report: ',classreport)


# In[13]:


#Gradient Boost Factors' Importance
xgbimportance=pd.DataFrame()
xgbimportance['variables']=x.columns
xgbimportance['importance']=xgb.feature_importances_
xgbimportance.sort_values(by='importance',ascending=False)


# In[14]:


df['predicted']=xgb.predict(x)
#accuracy=overall correctness
accuracy=accuracy_score(df['survived'],df['predicted'])
#precision=ability to correctly predict positive class
precision=precision_score(df['survived'],df['predicted'])
#recall=quantifies ability to correctly predict positive class
recall=recall_score(df['survived'],df['predicted'])
#confmatrix=[TP FP][FN TN]
confmatrix=confusion_matrix(df['survived'],df['predicted'])
classreport=classification_report(df['survived'],df['predicted'])
print('Accuracy: ',accuracy)
print('Precision: ',precision)
print('Recall: ',recall)
print('Confusion Matrix: ',confmatrix)
print('Classification Report: ',classreport)

