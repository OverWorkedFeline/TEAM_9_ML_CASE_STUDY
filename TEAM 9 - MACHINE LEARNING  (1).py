#!/usr/bin/env python
# coding: utf-8

# ## LEAD SCORING CASE STUDY (TEAM 9 SEPT 2022 MODULE EXAM)

# ### PROBLEM STATEMENT
# 
#     An education company named X Education sells online courses to our industry professionals.
#     many professionals who are interested in the courses land on their website for courses.
#     
#     The company is using social Media marketing on various websites and search engines like google
#     once the people land on websites, they might get intrested and browse courses or fill up a 
#     form for the course or watch videos.
#     1. Important to know that these people are classified as leads as they fill up a form 
#     providing their email address or phone number. 
#     Moreover, the company also gets leads through past referrals
#     2. Important to know that once the they get a lead. sales team contacts them.
#     3. Important to know that Thru this process, some of the leads get converted while most do 
#     not. TYPICAL LEAD CONVERSION RATE IS ABOUT 30%.
#     
#     X Education has now hired you!!!
#     They have given you a task.
#     1. SELECT THE MOST PROMISING LEADS. i.e. the leads that are most likely to convert into paying 
#     customers.
#     2. The company requires a model. wherein you need to assign a leadscore to each of the leads
#     such that the customers with higher lead socre have a higher conversion chance and the
#     customers with lower lead score havd a lower conversion chance.
#     3. The CEO, in particular has given a target lead conversion rate to be around 80%.
#     

# ### GOALS OF THE CASE STUDY
# 
#     1. Build a LOGISTIC REGRESSION MODEL to assign a lead score between 0 and 100 to each of the 
#     leads which can be used by the company to target potential sales.
#     2. higher the score would mean that the lead is hot.
#     hot here means the lead is most likely to convert, whereas a lower score would mean that
#     the lead is cold and will most likely skip.

# In[1]:


#basic modules
import pandas as pd
import numpy as np
#data Visualization modules
import matplotlib.pyplot as plt
import seaborn as sns
#supress warnings
import warnings
warnings.filterwarnings('ignore')
#increase view limits
pd.options.display.max_columns = None
pd.options.display.max_rows = 100
pd.options.display.float_format= '{:.2f}'.format


# ### Reading and Understanding Data

# In[2]:


# importing os 
import os
os.chdir(r"C:\Users\stanl\Downloads")


# In[3]:


#reading the data file using pandas.
df = pd.read_csv('Leads.csv')

df.head()


# In[4]:


# checking shape
df.shape

# 9240 rows and 37 columns


# In[5]:


# checking stats for numerical columns
df.describe()


# In[6]:


# checking dublicates.
df.duplicated().sum()

# No dublicate rows. duplicated function returns a bool of rows which contain False for no dublicates.


# In[7]:


# checking all columns, their datatypes and also get idea of amount of null values
df.info('all')


# #### Observations : 
#     1. alot of columns have alot of null values and are not so important, it is safe to drop it.
#     2. Prospect ID and Lead Number both have a unique idenfier(Primary key). we can drop either 
#     one of them. it is safe to drop Prospect ID.
#     3. Columns names are toooooooo long.
#     4. we have observed 'select' insted of 'Nan'. it means that select appears when someone does
#     not select anything from the dropdown in the form provided.
#     

# ### Data Cleaning
# 
# #### Rename column names.
#     1. Long column names makes analysis a bit hard.
#     2. convert into snake case (personal preference to handle data.)
#     

# In[8]:


# change into snake case (space to ____)

df.columns = df.columns.str.replace(' ','_').str.lower()

# checks

df.columns


# In[9]:


# shorten column name

df.rename(columns=
          {'totalvisits': 'total_visits',
           'total_time_spent_on_website': 'time_on_website', 
           'how_did_you_hear_about_x_education': 'source',
           'what_is_your_current_occupation': 'occupation',
           'what_matters_most_to_you_in_choosing_a_course' : 'course_selection_reason', 
           'receive_more_updates_about_our_courses': 'courses_updates', 
           'update_me_on_supply_chain_content': 'supply_chain_content_updates',
           'get_updates_on_dm_content': 'dm_content_updates',
           'i_agree_to_pay_the_amount_through_cheque': 'cheque_payment',
            'a_free_copy_of_mastering_the_interview': 'mastering_interview'},
          inplace = True)

# df.rename mein dict ka key value dalo. kush raho.
# inplace = True means it is overwriding changes to the actual data set.

# checking

df.columns.to_list()


# #### Drop prospect_id column

# In[10]:


df.drop(['prospect_id'],axis=1,inplace=True)


# #### Replace 'select' category with Nan or null values.
# 

# In[11]:


# select data types with object as datatype or non numeric columns
df_non_numeric=df.select_dtypes(include='object')

# find out columns who have'Select'
columns_with_select=df_non_numeric.columns[df_non_numeric.apply(lambda x : x.str.contains('Select',na=False)).any()]
columns_with_select.to_list()


# There are 4 columns that contain 'Select', which are effectively treated as null values.
# we are going to make that change.

# In[12]:


#replace values
select=columns_with_select.to_list()
df[select]=df[select].replace('Select',np.nan)


# #### Handle null values and columns generated by sales team / flags
# 
# 1. Given are alot of columns with very high number of null entities. best is to calculate the percentage of null values in the columns and take a decision.
# 2. We can drop sales related columns because they seem to be data entries that are made after the sales team has contacted the student / potential lead. those have no purpose of our model i.e providing lead scores
# 3. The columns are identified as :
#     - tages <br />
#     - lead_quality <br />
#     - last_activity <br />
#     - last_notable_activity <br />
#     - all columns which start with asymmetric name. <br />
#     

# In[13]:


# Percentage of null values for each column in df
(df.isnull().sum()/len(df))*100


# **Observation :** we can see that few columns with high percentage of missing data.
# since, there are no ways to get data back, we can drop this columns.

# #### Drop columns that have 40% more null values, sales generated flags or tags and columns named as assymetric 

# In[14]:


df.drop(
    [
    'source',
    'lead_quality',
    'lead_profile',
    'asymmetrique_activity_index',
    'asymmetrique_profile_index',
    'asymmetrique_activity_score',
    'asymmetrique_profile_score',
    'last_activity',
    'last_notable_activity'
    ],axis=1,inplace=True)

df.columns


# In[15]:


# we are left with few columns with nulls, rechecking
(df.isnull().sum()/len(df))*100


# **Observation** : we have 5 columns with high null values
# - country
# - specialization
# - occupation
# - course_selection_reason
# - city
# 
# we look at them individually to see what can be done.

# **Country Column**

# In[16]:


df.country.value_counts(normalize=True,dropna=False)*100


# In[17]:


plt.figure(figsize=(15,5))
s1=sns.countplot(df.country,hue=df.converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# **Observation :** This Distribution of the data is very heavily skewed with India + Nullvalues.
# It is safe to drop this column.

# In[18]:


df.drop('country',axis=1,inplace=True)


# **Column course_selection_reason**

# In[19]:


df.course_selection_reason.value_counts(normalize=True,dropna=False)*100


# **Observation :** The distribution of data is heavily skewed with Better Career Prospects + Nullvalues.
# It is safe to drop this column.

# In[20]:


df.drop('course_selection_reason',axis=1,inplace=True)


# **Occupation Column**

# In[21]:


df.occupation.value_counts(normalize=True,dropna=False)*100


# **Observation :** For Occupation, we can combine the categories and then impute.

# 29% Missing Values show that the column **who don't have any jobs didn't specify their Occupation**<br/>
# So we can Impute it with Mode as it is categorical column.

# In[22]:


df['occupation'].replace(np.nan, 'Unemployed',inplace=True)


# In[23]:


df.occupation.value_counts(normalize=True,dropna=False)*100


# In[24]:


# Visualizing the column after replacement

plt.figure(figsize=(15,5))
g1 = sns.countplot(df['occupation'], hue = df.converted)
plt.xticks(rotation=90)
plt.show()


# - Unemployed leads are the most in terms of absolute numbers
# - working professionals have a higher chance to get converted

# In[25]:


df.occupation.value_counts(normalize=True,dropna=False)*100


# **Specialization column**

# In[26]:


df.specialization.value_counts(normalize = True, dropna = False) * 100


# **Observation :** we can combine categories and impute them in propotion 
# Handling missing values.
# - replacing the missing values in Specialization column with 'Not Specified' as lead may not have mentioned specialization because it was not in the list or maybe they are a students and don't have a specialization yet. 
# - So we will replace NaN values here with 'Not Specified'

# In[27]:


df['specialization'].replace(np.nan,'Not Specified',inplace=True)


# In[28]:


df.specialization.value_counts(normalize = True) * 100


# **we can combine the management specialisations as they fall under the same category**

# In[29]:


df['specialization'].replace(['Finance Management','Human Resource Management','Marketing Management',
                                 'Operations Management','IT Projects Management','Supply Chain Management',
                                 'Healthcare Management','Hospitality Management','Retail Management'],
                                 'Management', inplace = True)  


# In[30]:


df.specialization.value_counts(normalize = True) * 100


# **City column**

# In[31]:


df.city.value_counts(normalize = True, dropna = False) * 100


# In[32]:


# Visualizing the country column

plt.figure(figsize=(15,5))
s1 = sns.countplot(df.city, hue = df.converted)
plt.xticks(rotation=90)
plt.show()


# **we can clearly see that Mumbai is the Mode of the Given Column** 
# - Replacing the missing values with Mode. 
# - As it is a categorical column we use mode

# In[33]:


df['city'].replace(np.nan,df['city'].mode()[0],inplace=True)


# In[34]:


df['tags'].value_counts(dropna=False)


# In[35]:


#replacing Nan values with "Not Specified"

df['tags'] = df['tags'].replace(np.nan,'Not Specified')


# #### Handle categorical columns with less missing values and low representation of categories
# 
# - Impute missing values.
# - Merge categories that have low representation.

# In[36]:


df['tags'] = df['tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized','switched off',
                                      'Already a student',
                                       'Not doing further education',
                                       'invalid number',
                                       'wrong number given',
                                       'Interested  in full time MBA'], 'Other_Tags')


# In[37]:


df.isnull().sum()/len(df)*100


# In[38]:


# determine all unique values for object datatype columns.
df.select_dtypes(include='object').nunique()


# **Observation :** As we can see that number of unique value is greater than 3 in two columns.
# - lead_source
# - lead_origin

# **Column lead_source**

# In[39]:


df.lead_source.value_counts(normalize=True,dropna=False)*100


# In[40]:


df.lead_source.mode()[0]


# In[41]:


# we can impute missing values with mode of data. i.e Google.
df.lead_source.fillna('Google',inplace=True)


# In[42]:


df['lead_source'] = df['lead_source'].replace('google','Google')
df['lead_source'] = df['lead_source'].replace('Facebook','Social Media')


# In[43]:


# There are alot of smaller values which will not be used. so we are safe to group them all.
df['lead_source'] = df['lead_source'].apply(lambda x: x if 
                                            ((x== 'Google') | (x=='Direct Traffic') | (x=='Olark Chat') | 
                                             (x=='Organic Search') | (x=='Reference')) 
                                            else 'Other Social Sites')


# In[44]:


df.lead_source.value_counts(normalize=True,dropna=False)*100


# In[45]:


# Visualizing the column after replacement
plt.figure(figsize=(15,5))
s1 = sns.countplot(df['lead_source'], hue = df.converted)
plt.xticks(rotation=90)
plt.show()


# **Observation**
# - Max Number of leads are generated by 'Google' and 'Direct Traffic'
# - Conversion rate for referals are higher!
# - we have to focus on 'Olark Chat' and other columns to convert to leads
# - other social sites do have a higher conversion rate.
# - maybe google is lower due to the fact that they first googled the course but got thru by some other means (googling for information)

# **column lead_origin**

# In[46]:


df.lead_origin.value_counts(normalize = True, dropna = False) * 100


# In[47]:


# Visualizing the Lead Origin column

plt.figure(figsize=(15,5))
s1 = sns.countplot(df['lead_origin'], hue = df.converted)
plt.xticks(rotation=90)
plt.show()


# **Observations**
# - API brings alot of people but less conversions
# - Landing page Submission are higher in number and more than 50 % conversions
# - very few leads from lead import and quick add form
# - Lead Add Form has higherconversion rate.
# - In order to improve overall lead conversion rate, we have to improve lead conversion of API and landing page submission.
# 

# #### Handle Bindary Columns
# 
# - Drop those Columns that have data imbalance
# - Drop those Columns that have only 1 unique entry

# In[48]:


df.select_dtypes(include='object').nunique()


# **Observation** 
# - The columns can be droped cause they have only one Unique value.
#     - cheque_payment
#     - courses_updates
#     - supply_chain_content_updates
#     - dm_content_updates
#     - magazine

# In[49]:


df.drop(['cheque_payment','courses_updates','supply_chain_content_updates','dm_content_updates','cheque_payment','magazine'],axis=1,inplace=True)


# **Lets check for data imbalance for the rest of the columns**

# In[50]:


# select rest of columns with 2 unique values
col=['do_not_email', 'do_not_call', 'search', 'newspaper_article', 'x_education_forums', 
           'newspaper', 'digital_advertisement', 'through_recommendations', 'mastering_interview']

df_binary=df[col]

for _ in df_binary.columns:
    print(df_binary[_].value_counts(normalize=True)*100)


# **Observation**
# 
# - The following columns can be droped as well. why, cause they have a heavy imbalance.
#     - through_recommendations
#     - newspaper
#     - x_education_forums
#     - digital_advertisement
#     - do_not_call
#     - search
#     - newspaper_article

# In[51]:


df.drop(['do_not_call', 'search', 'newspaper_article', 'x_education_forums', 
           'newspaper', 'digital_advertisement', 'through_recommendations','mastering_interview'],axis=1,inplace=True)


# #### Handle Numerical columns
# 
# ##### Column lead_number
# - lead_number is a unique identifier for each lead.<br/>
# - Aggretations won't be of any help. so change it to Object.

# In[52]:


df.lead_number=df.lead_number.astype('object')


# ##### Checking for Outliers.
# - In Numerical Columns. 
#     - total_visits
#     - time_on_website
#     - page_views_per_visit
# - Make a box plot to check Outliers.
# - we make box plot to check if we have to impute mean or median.

# In[53]:


# STYLESSSSSSS
plt.style.use('bmh')


# In[54]:


req_columns=['total_visits','time_on_website','page_views_per_visit']

plt.figure(figsize=(10,10))
for _ in enumerate(req_columns):
    plt.subplot(len(req_columns),1,_[0]+1)
    sns.boxplot(df[_[1]].dropna(),orient='h')


# **Presence of Outliers in _total_visits_ and _page_views_per_visit_ so we can impute Median Safely**
# 
# 1. So Handling Missing Data by imputing median
# 2. A general observation is that column _total_visits_ is a float. Since Total Visits cant be a float.
# 

# In[55]:


df.total_visits.fillna(df.total_visits.median(),inplace=True)
df.total_visits = df.total_visits.astype('int')


# In[56]:


df.page_views_per_visit.fillna(df.page_views_per_visit.median(),inplace=True)


# we can drop do not email column as it has not much significance.
# 

# In[61]:


df.drop('do_not_email',axis=1,inplace=True)


# ##### Data Cleaning Results.

# In[62]:


df.info()


# ### EDA (Exploratory Data Analysis)

# #### Numerical Columns

# In[58]:


xx
# Plot Stylesssssssssssss
plt.style.use('ggplot')


# In[ ]:


titles=['Total website visits','Total time spent on websites','Average number of page views per visit']
plt.figure(figsize=(20,20))
for _ in enumerate(req_columns):
    plt.subplot(len(req_columns),1,_[0]+1)
    plt.hist(df[_[1]].dropna(),bins=40)
    x=_[0]
    plt.title(titles[x])
    plt.show


# **Observations :** 
# - High peaks and Right Skewered Data.
# - Possibility of Outliers.
# - Have Checked them.

# #### Heatmap

# In[ ]:


plt.figure(figsize=(15,13))
sns.heatmap(df[req_columns].corr(),cmap='BuPu',annot=True)


# ##### No corelation in this columns so we don't need to drop them (positive Corelation)

# ### Categorical Columns

# #### Lead Origin

# In[ ]:


plt.style.use('dark_background')
plt.figure(figsize=(12,6))
x=df.groupby('lead_origin')
x['lead_number'].count().sort_values().plot(kind='barh',edgecolor='b',color=['#7f7f7f', '#bcbd22', '#17becf'])
plt.show()


# #### Lead Source

# In[ ]:


plt.style.use('dark_background')
plt.figure(figsize=(12,6))
x=df.groupby('lead_source')
x['lead_number'].count().sort_values().plot(kind='bar',edgecolor='b',color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd','#8c564b'])
plt.show()


# Most people come from 'Google'.

# #### Specialization

# In[ ]:


plt.style.use('dark_background')
plt.figure(figsize=(12,6))
x=df.groupby('specialization')
x['lead_number'].count().sort_values(ascending= False).plot(kind='bar',edgecolor='b')
plt.show()


# Most of the Specialization is coming from Management Professionals

# #### Occupation

# In[ ]:


plt.figure(figsize=(12,6))
x=df.groupby('occupation')
x['lead_number'].count().sort_values(ascending= False).plot(kind='bar',edgecolor='b')
plt.show()


# Unemployed users are more significant to be a lead.

# #### City

# In[ ]:


plt.figure(figsize=(12,6))
x=df.groupby('city')
x['lead_number'].count().sort_values(ascending= False).plot(kind='barh',edgecolor='b')
plt.show()


# Mumbai and Maharashtra in general dominates the markets.<br/>
# This is likely to the fact that courses are based in Mumbai??

# ### Data Prepration

# In[ ]:


df.select_dtypes(include='object').nunique()


# #### Converting Binary Columns

# In[ ]:


#converting yes / no in do_not_email
from sklearn.preprocessing import LabelEncoder

label= LabelEncoder() 
df['do_not_email'] = label.fit_transform(df['do_not_email'])
df['mastering_interview']=label.fit_transform(df['mastering_interview'])

#checking
df.head()


# Sucessfully converted Yes/No to 1/0 Using **Label Encoder**

# #### Creating Dummy Varaibles for Categorical columns
# 
# Categorical Columns are : lead_origin, lead_source, Specialization, occupation, city.

# In[ ]:


#create dummies
dumdum=pd.get_dummies(df[['lead_origin', 'lead_source', 'specialization', 'occupation', 'city','tags']],drop_first=True)

dumdum.head()

#concat results to original data frame

df=pd.concat([df,dumdum],axis=1)


# In[ ]:


#Droping the columns for which dummies are created.
df.drop(['lead_origin', 'lead_source', 'specialization', 'occupation', 'city','tags'],axis=1,inplace=True)


# In[ ]:


df.head()


# #### Handling Outliers.

# - First we check the 99% th values.
# - Then we can cap the outliers 
# - capping means replacing the outliers with .99%tile of the values. 
# - This is one of the methods to handle outliers just like imputation.

# In[ ]:


#checking .99 percentile of the values. for tota_visits
df['total_visits'].describe(percentiles=[.99])


# In[ ]:


#checking .99 percentile of the values. for page_views_per_visit
df['page_views_per_visit'].describe(percentiles=[.99])


# In[ ]:


#replacing values outliers.
df.total_visits.loc[df.total_visits>=df.total_visits.quantile(0.99)]=df.total_visits.quantile(0.99)
df.page_views_per_visit.loc[df.page_views_per_visit>=df.page_views_per_visit.quantile(0.99)]=df.page_views_per_visit.quantile(0.99)


# In[ ]:


plt.style.use('ggplot')
plt.figure(figsize=(10,14))

plt.subplot(2,1,1)
sns.boxplot(df.total_visits)

plt.subplot(2,1,2)
sns.boxplot(df.page_views_per_visit)

plt.show()


# Outliers have been reduced by capping/replacing it.

# In[ ]:


plt.style.use('dark_background')

titles=['Total website visits','Total time spent on websites','Average number of page views per visit']
plt.figure(figsize=(20,20))
for _ in enumerate(req_columns):
    plt.subplot(len(req_columns),1,_[0]+1)
    plt.hist(df[_[1]].dropna(),bins=10)
    x=_[0]
    plt.title(titles[x])
    plt.show


# ### Test-Train Split

# In[ ]:


#import test train library
from sklearn.model_selection import train_test_split


# In[ ]:


# Drop the converted column as we need to have that as dependent varaible.

x=df.drop(['converted','lead_number'],axis=1)

x.head()


# In[ ]:


# Dependent Varaible

y=df['converted']

y.head()


# In[ ]:


#Split the data set into 80% and 20% test train respectively.
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,test_size=0.3,random_state=10)


# **Obesrvation from EDA.**
# - The Numerical columns are Right skewered.
# - The Best way to Scale this kind of distribution is to perform a MinMax Feature scaling.

# In[ ]:


#import minmax scaler 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train[['total_visits','time_on_website','page_views_per_visit']]=scaler.fit_transform(x_train[['total_visits','time_on_website','page_views_per_visit']])
x_test[['total_visits','time_on_website','page_views_per_visit']]=scaler.fit_transform(x_test[['total_visits','time_on_website','page_views_per_visit']])


# In[ ]:


x_test.drop(['lead_origin_Lead Add Form and Others', 'specialization_Industry Specializations', 
                     'occupation_Working Professional'], axis = 1, inplace = True)

x_train.drop(['lead_origin_Lead Add Form and Others', 'specialization_Industry Specializations', 
                     'occupation_Working Professional'], axis = 1, inplace = True)


# ### Model Building

# In[ ]:


#Importing Necessary Library
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor


# #### RFE (Recursive feature Elemination)
# - There are a few ways to remove features in models.
# - Either thru one by one elimination or by using techniques provided by Sklearn.
# - RFE removes Features one by one until the optimal number of features is left.
# - This is the most important feature selection algorithms due to ease of use and its flexibility.
# - The algorithm can wrap around any model, and it produces the best possible set of features that gives the highest performance.
# - RFE will give the same accuracy with less number of Features.

# In[ ]:


# initiate logistic regression
logR = LogisticRegression()

# initiate rfe
rfe = RFE(logR, n_features_to_select=16)             # running RFE with 13 variables as output
rfe = rfe.fit(x_train, y_train)


# We fit the estimator into the RFE class.
# the rfe.support_ attribute that gives a boolean mask with False values for discarded features.
# we can use it to subset our data.

# In[ ]:


rfe.support_


# **The Columns that RFE selects are down below**

# In[ ]:



col = x_train.columns[rfe.support_]
print(col)


# In[ ]:


x_train.columns[~rfe.support_]


# **The columns that RFE don't select are above**

# In[ ]:


list(zip(x_train.columns, rfe.support_, rfe.ranking_))


# **RFE Ranking the column revelence above**

# ### **FIRST MODEL** (Made using Feature Selection)

# **Model Stats Using statsmodels**

# In[ ]:


#model 1
x_train_sm = sm.add_constant(x_train[col])

log1 = sm.GLM(y_train,x_train_sm, family = sm.families.Binomial())
res = log1.fit()
res.summary()


# **Removing tags_Interested in Next batch,tags_Interested in other courses as their P-Value is very High**

# In[ ]:


col=col.drop(['tags_Interested in Next batch','tags_Interested in other courses'])
col=col.drop(['lead_origin_Landing Page Submission'])


# ### **SECOND MODEL** (Made using Feature Selection)

# In[ ]:


#model 2
x_train_sm = sm.add_constant(x_train[col])

logm2 = sm.GLM(y_train,x_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = x_train[col].columns
vif['VIF'] = [variance_inflation_factor(x_train[col].values, i) for i in range(x_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# **CHECKING VARIENCE INFLUENCE FACTORS**
# 
# - p-value for city_Non-Maharashtra Cities is high but the varience factor is low. keeping column
# - p-value for tags_Ringing is high but the varience factor is low. keeping column

# In[ ]:


y_train_pred = res.predict(x_train_sm).values.reshape(-1)
y_train_pred


# In[ ]:


y_train_pred_final = pd.DataFrame({'Converted': y_train.values, 'Converted_Prob': y_train_pred,'Lead_score':y_train_pred*100})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[ ]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()
from sklearn import metrics

print(metrics.accuracy_score(y_true= y_train_pred_final.Converted, y_pred= y_train_pred_final.Predicted))


# In[ ]:


confusion = metrics.confusion_matrix(y_true= y_train_pred_final.Converted, y_pred= y_train_pred_final.Predicted)
confusion


# In[ ]:




