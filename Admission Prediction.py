#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV,ElasticNet,ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


# In[86]:


import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter


# In[2]:


data_ap=pd.read_csv(r"C:\Users\dell\Downloads\Admission_Prediction.csv")


# In[3]:


data_ap.head()


# In[4]:


from pandas_profiling import ProfileReport


# In[149]:


pf=ProfileReport(data_ap)


# In[150]:


pf


# In[151]:


#save the report file
pf.to_file("Admission_Prediction_LR.html")


# In[6]:


pf.to_widgets()


# In[8]:


data_ap.describe()


# In[9]:


data_ap.isnull().sum()


# In[10]:


data_ap["GRE Score"] = data_ap["GRE Score"].fillna(data_ap['GRE Score'].mean())


# In[11]:


data_ap["TOEFL Score"] = data_ap["TOEFL Score"].fillna(data_ap['TOEFL Score'].mean())


# In[12]:


data_ap["University Rating"] = data_ap["University Rating"].fillna(data_ap['University Rating'].mean())


# In[36]:


data_ap.describe()


# In[14]:


data_ap.isnull().sum()


# In[37]:


#Drop 
data_ap.drop(columns=["Serial No."],inplace=True)


# In[42]:


data_ap


# In[43]:


#Label column
label_col=data_ap['Chance of Admit']


# In[44]:


label_col


# In[45]:


#Feature column
feature_col=data_ap.drop(columns=['Chance of Admit'])


# In[46]:


feature_col


# # standardization

# In[47]:


#standardrization
#removes mean of the data
#it help the model to understand the data well


# In[48]:


scalar=StandardScaler()
"""
If the dispersion of the data is more ,StandardScaler is used to mitigate the same. bring data to normal distribution"""


# In[49]:


new_ftr=scalar.fit_transform(feature_col)


# In[50]:


new_ftr


# In[51]:


new_ftr1=pd.DataFrame(new_ftr)


# In[52]:


new_ftr1


# In[53]:


new_ftr1.profile_report()


# In[54]:


new_ftr1.describe()


# In[55]:


new_ftr1


# In[56]:


#checking multicollinearity
#import VIF(variance inflation factor=1/(1-r^2))
from statsmodels.stats.outliers_influence import variance_inflation_factor

#variance_inflation_factor()


# In[59]:


#VIF
[variance_inflation_factor(new_ftr1,i) for i in range(new_ftr1.shape[1])]


# In[60]:


vif_df=pd.DataFrame()


# In[61]:


vif_df['vif']=[variance_inflation_factor(new_ftr1,i) for i in range(new_ftr1.shape[1])]


# In[62]:


vif_df['Feature'] = feature_col.columns


# In[63]:


vif_df


# In[64]:


"""
There is no multicollinearity as the value of vif < 10.

"""


# # Model building

# TRAIN TEST AND SPLIT

# In[94]:


x_train,x_test,y_train,y_test=train_test_split(new_ftr1,label_col,test_size=0.15,random_state=100)


# In[95]:


x_train


# In[96]:


lin_reg=LinearRegression()


# In[97]:


lin_reg.fit(x_train,y_train)


# # Save Model

# In[98]:


import pickle


# In[99]:


pickle.dump(lin_reg,open('admission_lin_reg_model.pickle','wb'))


# In[100]:


#check the created file.
ls


# In[101]:


#prediction
lin_reg.predict([[314.000000,103.0,2.0,2.0,3.0,8.21,0]])


# In[102]:


"""
prediction result is not as expected it should be 6.5 but comming 7.465.
it happend we have build the model on the transfom data (standardization).
so to get the correct output input should be transform.
"""


# In[103]:


test_ap=scalar.transform([[314.000000,103.0,2.0,2.0,3.0,8.21,0]])


# In[147]:


test_ap


# # Lasso Model

# In[104]:


#prediction
lin_reg.predict(test_ap)


# In[105]:


"""Now it is close to the desire result."""


# In[106]:


model=pickle.load(open('admission_lin_reg_model.pickle','rb'))


# In[107]:


#predicting model from the model file saved.
model.predict(test_ap)


# In[108]:


#Accuracy
#R^2
lin_reg.score(x_test,y_test)


# In[109]:


#adjusted R^2
def adjs_r2(x,y):
    r2=lin_reg.score(x,y)
    n=x.shape[0]
    p=x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# In[110]:


adjs_r2(x_test,y_test)


# In[111]:


#Getting the 'm' and 'c' values of the model equation.


# In[112]:


#'m'
lin_reg.coef_


# In[113]:


#'c'
lin_reg.intercept_


# In[ ]:





# In[ ]:





# In[118]:


lassocv=LassoCV(cv=10,max_iter=2000000, normalize = True)
#cv= 5 combination of the test and train data set(5_set)
lassocv.fit(x_train,y_train)


# In[119]:


lassocv.alpha_


# In[120]:


lasso= Lasso(alpha=lassocv.alpha_)
lasso.fit(x_train,y_train)


# In[121]:


lasso.score(x_test,y_test)


# 
# # RIDGE_model

# In[131]:


ridgecv= RidgeCV(alphas=np.random.uniform(0,10,50),cv=10,normalize=True)
ridgecv.fit(x_train,y_train)


# In[134]:


ridge_lr=Ridge(alpha=ridgecv.alpha_)
ridge_lr.fit(x_train,y_train)


# In[135]:


ridge_lr.score(x_test,y_test)


# # ELASTIC NET_Model

# In[136]:


elastic=ElasticNetCV(alphas=None, cv=10)
elastic.fit(x_train,y_train)


# In[137]:


elastic.alpha_


# In[139]:


elastic_lr=ElasticNet(alpha=elastic.alpha_,l1_ratio=elastic.l1_ratio_)


# In[140]:


elastic_lr.fit(x_train,y_train)


# In[141]:


elastic.score(x_test,y_test)


# In[ ]:




