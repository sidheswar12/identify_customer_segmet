#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[2]:


# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv',sep=';')

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv',sep=';')


# In[3]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).
azdias.head()


# In[4]:


azdias.sample(10)


# In[5]:


# Structure of dataframe; followed by investigation cells
azdias.shape
print('Number of rows:', azdias.shape[0])
print('Number of columns', azdias.shape[1])


# In[6]:


azdias.info()


# In[7]:


azdias['AGER_TYP'].unique()


# In[8]:


azdias.describe().transpose()


# In[9]:


# Look at entire feat_info dataframe
feat_info.head(85)


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[10]:


# Identify missing or unknown data values and convert them to NaNs.
azdias.isnull().any()


# In[11]:


azdias.isnull().sum()


# In[12]:


# Total number of naturally missing observations
sum(azdias.isnull().sum())


# In[13]:


# Visualizing the naturally missing data
azdias.isnull().sum().plot.bar(figsize=(20,8),fontsize=12,color='teal');


# In[14]:


# Deeper analysis of the variable KK_KUNDENTYP - Consumer pattern over 12 months
azdias['KK_KUNDENTYP'].value_counts()


# In[15]:


# Identify missing or unknown data values and convert them to NaNs.
def convert_missing_to_nan(df):
    for i,V in enumerate(df.iteritems()):
        missing_unknown = feat_info['missing_or_unknown'][i]
        column_name = V[0]
        missing_unknown = missing_unknown[1:-1].split(',')
        if missing_unknown != ['']:
            hold = []
            for x in missing_unknown:
                if x in ['X','XX']:
                    hold.append(x)
                else:
                    hold.append(int(x))
            df[column_name] = df[column_name].replace(hold,np.nan)
            
    return df


# In[16]:


azdias = convert_missing_to_nan(azdias)


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[17]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.
azdias.isnull().sum()


# In[18]:


sum(azdias.isnull().sum())


# In[19]:


#Visualizing the columns with missing values
azdias.isnull().sum().plot.bar(figsize=(20,8),fontsize=12,color='teal');


# In[20]:


# Visualizing the missing data for each column sorted from highest to lowest.
azdias.isnull().sum().sort_values().plot.barh(figsize=(20,30),fontsize=12,color='teal');


# In[21]:


# Calculating percentage of Nan values in descending order
round(azdias.isnull().sum()/azdias.shape[0]*100,1).sort_values(ascending=False)


# In[22]:


# Investigate patterns in the amount of missing data in each column.
plt.figure(1,figsize=(20,8))
plt.subplot(111)
azdias.loc[:,'AGER_TYP':'SEMIO_TRADV'].isnull().sum().plot.bar(fontsize=12,color='green')
plt.title('Person-Level Features',fontsize=15)

plt.figure(2,figsize=(20,6))
plt.subplot(131)
azdias.loc[:,'ALTER_HH':'WOHNDAUER_2008'].isnull().sum().plot.bar(fontsize=12,color='red')
plt.title('Household-Level Features',fontsize=15)

plt.subplot(132)
azdias.loc[:,'ANZ_HAUSHALTE_AKTIV':'WOHNLAGE'].isnull().sum().plot.bar(fontsize=12,color='orange')
plt.title('Building-Level Features',fontsize=15)

plt.subplot(133)
azdias.loc[:,'BALLRAUM':'INNENSTADT'].isnull().sum().plot.bar(fontsize=12,color='skyblue')
plt.title('Postcode-Level Features',fontsize=15)

plt.figure(3,figsize=(20,6))
plt.subplot(131)
azdias.loc[:,'CAMEO_DEUG_2015':'CAMEO_INTL_2015'].isnull().sum().plot.bar(fontsize=12,color='red')
plt.title('RR4 Micro-cell Features',fontsize=15)

plt.subplot(132)
azdias.loc[:,'KBA05_ANTG1':'KBA05_GBZ'].isnull().sum().plot.bar(fontsize=12,color='orange')
plt.title('RR3 Micro-cell Features',fontsize=15)

plt.subplot(133)
azdias.loc[:,'KBA13_ANZAHL_PKW':'PLZ8_GBZ'].isnull().sum().plot.bar(fontsize=12,color='skyblue')
plt.title('PLZ8 Macro-cell Features',fontsize=15)

plt.figure(4,figsize=(10,5))
plt.subplot(121)
azdias.loc[:,'GEBAEUDETYP_RASTER':'REGIOTYP'].isnull().sum().plot.bar(fontsize=12,color='red')
plt.title('RR1 Region Features',fontsize=14)

plt.subplot(122)
azdias.loc[:,'ARBEIT':'RELAT_AB'].isnull().sum().plot.bar(fontsize=12,color='orange')
plt.title('Community-Level Features',fontsize=14)

plt.show()


# In[23]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)

azdias = azdias.drop(['TITEL_KZ','AGER_TYP','KK_KUNDENTYP','KBA05_BAUMAX','GEBURTSJAHR','ALTER_HH'],axis=1)
azdias[:10]


# In[24]:


azdias.shape


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# 
# From the analysis of the naturally missing data, I can see that the person-level features is the most complete with the least number of missing observations. The data seems to have been collected and assembled by feature level, as 5 of the 9 feature levels contain the same number of missing observations, namely RR4 Micro-cell has 98,979 missing observations, RR3 Micro-cell 133,324, Postcode-level 93,740, PLZ8 Macro-cell 116,515 and Community-level 97,216 missing observations. In the person-level features, there are 9 columns with 4,854 observations missing.
# 
# "KK_Kundentyp" has the greatest number of observations missing at 584,612 observations or 66%. This category relates to consumer pattern over the past 12 months, so the presumption is missing observations were people from the population who were not a customer of the mail order company in the last 12 months. I think this column can de dropped from the dataset as the information is not useful in obtaining customer segments from the population. This information is part of the customer dataset. Decision drop "KK_Kundentyp".
# 
# Once the function to convert missing values from the feature summary file is added to the dataset, we have an increase in missing observations of 71% from 4,896,838 to 8,373,929. When we graph this data we can see 6 columns clearly stand out, all with over 30% missing information. Looking at each of the columns individually to determine their significance; "KK_Kundentyp" was discussed above. "TITEL_KZ" relates to an academic title and with 99.8% missing observations, this is a no brainer but to drop the column. "AGER_TYP" relates to a split in elderly persons, this information is partially found in other categories and can be also safely dropped without losing value. "KBA05_BAUMAX" is a summary of categories 5.1,5.2,5.3,5.4 and so can be safely dropped. "GEBURTSJAHR" relates to year of birth, since this information is genaralized in other person-level categories and with 44% of the information missing, imputing wouldn't make sense so I think better to drop also this column. Lastly "ALTER_HH" is again related to the date of birth of the head of the household and can be dropped for the same reasons as "GEBURTSJAHR".
# 
# Decision drop the 6 outlier columns of "TITEL_KZ", "AGER_TYP", "KK_KUDENTYP", "KBA05_BAUMAX", "GEBURTSJAHR" and "ALTER_HH".

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[25]:


# How much data is missing in each row of the dataset?
missing_row_data = azdias.isnull().sum(axis=1)
missing_row_data


# In[26]:


# Aggregate missing row data by number of missing values per row
missing_row_data.value_counts().sort_index()


# In[27]:


# Visualize counts of number of missing data points by row
missing_row_data.plot(kind='hist',bins=30,figsize=(20,8),color='teal');


# In[28]:


# Zoom in on missing rows from 10 missing data points
missing_row_data.plot(kind='hist',bins=30,figsize=(20,8),color='teal',xlim=10,ylim=(0,50000));


# In[29]:


#Zoom in on area from 10 to 25 missing data points
missing_row_data.plot(kind='hist',bins=30,figsize=(20,8),color='teal',xlim=(10,25),ylim=(0,10000));


# In[30]:


# Calculate percentage of data kept for rows with 9 or less missing data points
cumsum = round(missing_row_data.value_counts().sort_index().cumsum()[:9]/missing_row_data.shape[0]*100,1)
print("Percentage of data kept:",cumsum[8],"%")


# In[31]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.

azdias_subset_1 = azdias[azdias.index.isin(missing_row_data[missing_row_data <= 9].index)]
azdias_subset_1


# In[32]:


azdias_subset_1.shape


# In[33]:


azdias_subset_2 = azdias[azdias.index.isin(missing_row_data[missing_row_data > 9].index)]
azdias_subset_2


# In[34]:


azdias_subset_2.shape


# In[35]:


# Check all original rows have been accounted for and no data is missing
azdias_subset_1.shape[0] + azdias_subset_2.shape[0]


# In[36]:


# Header list names for reference for analysis between dataframes azdias subset 1 and 2
col_comparison = pd.Series(list(azdias_subset_1.columns.values))


# In[37]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.
def countplot(col1,col2,col3,col4,col5,col6):
    
    plt.figure(1,figsize=(16,4))
    
    plt.subplot(141)
    sns.countplot(x=azdias_subset_1[col1],color='skyblue',edgecolor='.9',data=azdias_subset_1)
    plt.title("Data Kept")

    plt.subplot(142)
    sns.countplot(x=azdias_subset_2[col1],color='lightgreen',edgecolor='.9',data=azdias_subset_2)
    plt.title("Data Rejected")
    
    plt.subplot(143)
    sns.countplot(x=azdias_subset_1[col2],color='skyblue',edgecolor='.9',data=azdias_subset_1)
    plt.title("Data Kept")

    plt.subplot(144)
    sns.countplot(x=azdias_subset_2[col2],color='lightgreen',edgecolor='.9',data=azdias_subset_2)
    plt.title("Data Rejected")
    
    plt.subplots_adjust(wspace = 0.5)
    
    plt.figure(2,figsize=(16,4))
    
    plt.subplot(141)
    sns.countplot(x=azdias_subset_1[col3],color='skyblue',edgecolor='.9',data=azdias_subset_1)
    plt.title("Data Kept")

    plt.subplot(142)
    sns.countplot(x=azdias_subset_2[col3],color='lightgreen',edgecolor='.9',data=azdias_subset_2)
    plt.title("Data Rejected")
    
    plt.subplot(143)
    sns.countplot(x=azdias_subset_1[col4],color='skyblue',edgecolor='.9',data=azdias_subset_1)
    plt.title("Data Kept")

    plt.subplot(144)
    sns.countplot(x=azdias_subset_2[col4],color='lightgreen',edgecolor='.9',data=azdias_subset_2)
    plt.title("Data Rejected")
    
    plt.subplots_adjust(wspace = 0.5)
    
    plt.figure(3,figsize=(16,4))
    
    plt.subplot(141)
    sns.countplot(x=azdias_subset_1[col5],color='skyblue',edgecolor='.9',data=azdias_subset_1)
    plt.title("Data Kept")

    plt.subplot(142)
    sns.countplot(x=azdias_subset_2[col5],color='lightgreen',edgecolor='.9',data=azdias_subset_2)
    plt.title("Data Rejected")
    
    plt.subplot(143)
    sns.countplot(x=azdias_subset_1[col6],color='skyblue',edgecolor='.9',data=azdias_subset_1)
    plt.title("Data Kept")

    plt.subplot(144)
    sns.countplot(x=azdias_subset_2[col6],color='lightgreen',edgecolor='.9',data=azdias_subset_2)
    plt.title("Data Rejected")
    
    
    plt.subplots_adjust(wspace = 0.5)
    plt.show();


# In[38]:


# Columns chosen manually with zero or almost zero missing data 
countplot('ANREDE_KZ','GREEN_AVANTGARDE','FINANZTYP','SEMIO_FAM','HH_EINKOMMEN_SCORE','LP_STATUS_FEIN')


# In[39]:


# 6 columns chosen at random for comparison
s = list(col_comparison.sample(n=6))
countplot(s[0],s[1],s[2],s[3],s[4],s[5])


# In[40]:


# Checkpoint of azdias dataset taken
azdias_subset_1.to_csv('azdias_checkpoint_1.csv',sep=';',encoding='utf-8',index=False)


# In[41]:


# Load from checkpoint
azdias_2 = pd.read_csv('azdias_checkpoint_1.csv',sep=';')
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv',sep=';')


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# First of all, we can see 623,209 rows are complete out of a total of 891,221 or 69.9%.
# 
# From the plots, we can see common missing number of observations in rows are 34, 45 and 47 missing observations with a combined total of 83,763 rows or 9.4%. With so many missing values I think the reliability and usefulness of these observations are limited, and the best decision is to drop them.
# 
# So zooming in on the plot we can see some natural breaks into the data at missing number of values at 9, 11, 17 and 20. The cumulative percentage of data retained at each of these cut off points are respectively; 9:86.6%, 11:87.7%, 17:89.3% and 20:89.5%. This means a difference of keeping an extra 25,376 rows between 9 missing observations on a row and 20 missing observations on a row.
# 
# Considering that the clustering algorithm K-means requires no missing information to work and that the dataset is made up of mainly ordinal and categorical data, where imputing values with means, modes or medians can impact the truth of the dataset much more than if the dataset was made up of numerical data. On balance, it seems more appropriate to minimize imputing missing data and maintain the integrity at maximum.
# 
# Therefore, the decision is to delete all rows with more than 9 missing observations on the row, thus keeping 774,743 rows (86.6%) and discarding the remaining 116,478 rows.
# 
# "It was a tough decision between discarding all rows with missing observations, therefore keeping 70% of the data but no imputation and the 9 missing rows, so keeping 87% of the data but imputing 148,492 values on the datset. To be honest in a real-world scenario I think I would have discarded all rows with any missing observations, on the basis that the German population over 18 is around 70 million. Therefore the difference between the two sample choices on the overall population is 0.2%. But I'm doing this course to learn so I want to do all the steps and practice everything."
# 
# With regards the data discarded with more than 9 missing observations. At first glance, the distributions within the variables in the most part seem different. Looking more closely at some of the variables manually chosen that have no missing observations only the gender "ANREDE_KZ" seems comparable, if you look at "HH_EINKOMMEN_SCORE" which shows an household net income the overwhelming majority of rows rejected show a 2 or very high income. This is also reflected in "LP_STATUS_FEIN" where the overwhelming majority of rows rejected are a 5 or minimalistic high-income earners. The missing observations in rows could relate to the extra privacy laws that protect high earners due the ease of being able to identify them from information. This would need more investigation to confirm.
# 
# Other categories in the rejected data are also highly different in distribution, for example "RETOURTYP_BK_S", "SEMIO_FAM", "SEMIO_KAEM" and "SEMIO_DOM", and so we can conclude that the discarded rows are qualitatively different to the rows that have been kept.

# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[42]:


# How many features are there of each data type?

print(feat_info['type'].value_counts())


# In[43]:


# Look at the categorical features
feat_info[feat_info['type'] == 'categorical']


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[44]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?

# Remove from feat_info dataframe colums that have already been deleted
feat_info = feat_info[feat_info.attribute != 'TITEL_KZ']
feat_info = feat_info[feat_info.attribute != 'AGER_TYP']
feat_info = feat_info[feat_info.attribute != 'KK_KUNDENTYP']
feat_info = feat_info[feat_info.attribute != 'KBA05_BAUMAX']
feat_info = feat_info[feat_info.attribute != 'GEBURTSJAHR']
feat_info = feat_info[feat_info.attribute != 'ALTER_HH']


# In[45]:


# Create a dataframe with only categorical variables
categorical_variables = azdias_2[feat_info[feat_info['type'] == 'categorical']['attribute']]
categorical_variables


# In[46]:


# List for reference of categorical variables
categorical_variables.info()


# In[47]:


# Separate the categorical variables into binary or multi-variable
binary = []
multivar = []
for x in categorical_variables:
    if len(categorical_variables[x].value_counts()) == 2:
        binary.append(x)
    else:
        multivar.append(x)


# In[48]:


# Binary list
binary


# In[49]:


# Multi-variable list
multivar


# In[50]:


# Find binary categorical variable non-numeric
azdias_2['ANREDE_KZ'].unique()


# In[51]:


# Find binary categorical variable non-numeric
azdias_2['GREEN_AVANTGARDE'].unique()


# In[52]:


# Find binary categorical variable non-numeric
azdias_2['SOHO_KZ'].unique()


# In[53]:


# Find binary categorical variable non-numeric
azdias_2['VERS_TYP'].unique()


# In[54]:


# Find binary categorical variable non-numeric - located it
azdias_2['OST_WEST_KZ'].unique()


# In[55]:


# Change 'OST_WEST_KZ into numerical values
azdias_2['OST_WEST_KZ'] = azdias_2['OST_WEST_KZ'].map({'W': 1, 'O': 0})


# In[56]:


# Analysis of column "GEBAEUDETYP"
azdias_2['GEBAEUDETYP'].value_counts().sort_index()


# In[57]:


# Simplify 'GEBAEUDETYP' Type of building use into 3 categories; 1 = residential, 2 = mixed usage, 3 = commercial
azdias_2['GEBAEUDETYP'] = azdias_2['GEBAEUDETYP'].map({1:1,2:1,3:2,4:2,5:3,6:2,7:3,8:2})


# In[58]:


# Check mapping has worked - After reviewing these numbers decided to drop as provide no added value information 
# for customer segmentation.
azdias_2['GEBAEUDETYP'].value_counts().sort_index()


# In[59]:


# Deep analysis of multi-variable category CAMEO_DEU_2015
azdias_2['CAMEO_DEU_2015'].value_counts().sort_index()


# In[60]:


# Drop multi-variable columns not required
cols_to_drop = ['LP_FAMILIE_GROB','LP_STATUS_GROB','GEBAEUDETYP','CAMEO_DEU_2015']
azdias_2 = azdias_2.drop(cols_to_drop,axis=1)


# In[61]:


# Re-encode categorical variable(s) to be kept in the analysis.

multi = []

for x in multivar:
    if x not in cols_to_drop:
        multi.append(x)
        
multi


# In[62]:


# One Hot Encode remaining multi-variable categorical variables
azdias_2 = pd.get_dummies(azdias_2, columns = multi, prefix=multi)


# In[63]:


azdias_2.head()


# In[64]:


azdias_2.shape


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# For the categorical features, I first isolated which columns classed as categorical and separated them between binary variables i.e. ones with 2 values and multiple variables.
# 
# The 5 binary categorical features, all give information that I think could be relevant for forming customer segments so the decision was to keep them all including the non-numerical one which related to whether the building in which the individual lives is in what was the old East or West Germany prior to 1991. After 27 years this feature is probably waning in usefulness compared to 10 or 20 years ago when analysing customer segments.
# 
# For the multi categorical variables, I analysed each individually: The following categories ("CJT_GESAMTTYP", "FINANZTYP", "NATIONALITAET_KZ", "SHOPPER_TYP","ZABEOTYP") all related to person-level features without being too detailed so the decision was to keep them and one hot encode them.
# 
# For "LP_FAMILIE_FEIN","LP_FAMILIE_GROB","LP_STATUS_FEIN", and "LP_STATUS_GROB', the decision was to drop the Grobs which were summaries of the Feins. I thought the level of segregation was better for identifying customer segments in the more detailed Fein scales but it was unnecessary to keep both.
# 
# The category "GEBAEUDETYP" once summarized provided no additional value for customer segmentation as it related to building-level feature which could be inferred from 3.1 "ANZ_HAUSHALTE_AKTIV" so it was decided to drop it.
# 
# Between "CAMEO_DEUG_2015", and "CAMEO_DEU_2015", which both relate to wealth/life stage typology with the first using a rough scale and the second a very detailed marketing scale. I decided to keep only the rough scale. When you look through all the categories in the feature info list, the majority are at a higher level in detail to "CAMEO_DEU_2015". "CAMEO_DEU_2015" seems to be a level of detail too high for this particular customer segment analysis, I think it would be appropriate to use this in a second analysis after primary customer segments have been identified for fine-tuning marketing efforts. Also there are many categories and one hot encoding these would add many features to our analysis and I am not convinced it would add value to this particular segment analysis.
# 
# Lastly, "GFK_URLAUBERTYP" is a detailed analysis of vacation habits with 12 categories. It was a close choice on whether to keep or drop for the same reasons discussed in the paragraph above on the level of detail for this particular analysis. On the balance of number of categories and that it was a person-level feature I have kept it and one hot encoded it.

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[65]:


# Identify features classified as mixed
feat_info[feat_info['type'] == 'mixed']


# In[66]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
azdias_2['PRAEGENDE_JUGENDJAHRE'].value_counts()


# In[67]:


# Engineer a column "AVANTGARDE" 
avantgarde = []

for x in azdias_2['PRAEGENDE_JUGENDJAHRE']:
    if x in [1,3,5,8,10,12,14]:
        avantgarde.append(0) # Value 0 denotes 'Mainstream'
    elif x in [2,4,6,7,9,11,13,15]:
        avantgarde.append(1) # Value 1 denotes 'Avantgarde'
    else:
        avantgarde.append(np.nan)


# In[68]:


# Add new column "AVANTGARDE" to azdias dataframe
a = pd.Series(avantgarde)
azdias_2['AVANTGARDE'] = a.values
azdias_2.head()


# In[69]:


# Check engineering successful
azdias_2['AVANTGARDE'].value_counts()


# In[70]:


# Engineer new column "DEKADE" based on decade of birth
azdias_2['DEKADE'] = azdias_2['PRAEGENDE_JUGENDJAHRE'].map({1:1,2:1,3:2,4:2,5:3,6:3,7:3,8:4,9:4,10:5,11:5,12:5,13:5,14:6,15:6})


# In[71]:


# Check engineering
azdias_2['DEKADE'].value_counts().sort_index()


# In[72]:


# Drop original column from dataframe
azdias_2 = azdias_2.drop('PRAEGENDE_JUGENDJAHRE',axis=1)


# In[73]:


# Check dataframe
azdias_2[:10]


# In[74]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.

azdias_2['CAMEO_INTL_2015'].value_counts()


# In[75]:


# Engineer a feature based on household wealth
cameo_wealth = []

for x in azdias_2['CAMEO_INTL_2015']:
    if 10 < x < 16:
        cameo_wealth.append(5)
    elif 20 < x < 26:
        cameo_wealth.append(4)
    elif 30 < x < 36:
        cameo_wealth.append(3)
    elif 40 < x < 46:
        cameo_wealth.append(2)
    elif 50 < x < 56:
        cameo_wealth.append(1)
    else:
        cameo_wealth.append(np.nan)


# In[76]:


cw = pd.Series(cameo_wealth)
azdias_2['CAMEO_WEALTH'] = cw.values


# In[77]:


azdias_2['CAMEO_WEALTH'].value_counts().sort_index()


# In[78]:


# Engineer a feature based on household lifestage
cameo_lifestage = []

for x in azdias_2['CAMEO_INTL_2015']:
    if x % 10 == 1:
        cameo_lifestage.append(1)
    elif x % 10 == 2:
        cameo_lifestage.append(2)
    elif x % 10 == 3:
        cameo_lifestage.append(3)
    elif x % 10 == 4:
        cameo_lifestage.append(4)
    elif x % 10 == 5:
        cameo_lifestage.append(5)
    else:
        cameo_lifestage.append(np.nan)


# In[79]:


cl = pd.Series(cameo_lifestage)
azdias_2['CAMEO_LIFESTAGE'] = cl.values


# In[80]:


azdias_2['CAMEO_LIFESTAGE'].value_counts().sort_index()


# In[81]:


# Remove original column
azdias_2 = azdias_2.drop(['CAMEO_INTL_2015'],axis=1)


# In[82]:


# Verify re-engineering of "CAMEO_INTL_2015
azdias_2.head()


# In[83]:


azdias_2['WOHNLAGE'].value_counts().sort_index()


# In[84]:


# Engineer a feature based on rural or not
azdias_2['RURAL'] = azdias_2['WOHNLAGE'].map({0:0,1:0,2:0,3:0,4:0,5:0,7:1,8:1})


# In[85]:


# Engineer a feature based on quality of neighborhood
azdias_2['GEGEND'] = azdias_2['WOHNLAGE'].map({0:0,1:1,2:2,3:3,4:4,5:5,7:0,8:0})


# In[86]:


# Drop mixed features not required
azdias_2 = azdias_2.drop(['LP_LEBENSPHASE_FEIN','LP_LEBENSPHASE_GROB','WOHNLAGE','PLZ8_BAUMAX'],axis=1)


# In[87]:


azdias_2.shape


# In[88]:


# Check dataframe features
azdias_2.head()


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# The 2 variables "PRAEGENDE_JUGENDJAHRE" and "CAMEO_INTL_2015" were re-engineered into 4 new features as required at the start of this section. ("AVANTGARDE", "DEKADE", "CAMEO_WEALTH" and "CAMEO_LIFESTAGE")
# 
# The "CAMEO_INTL_2015" was interesting as the wealth feature engineered is very similar to the "CAMEO_DEUG_2015" categorical feature one hot encoded in the previous section. I thought about whether both features were necessary. After consideration I concluded that even though the "CAMEO_INTL_2015" was created from the German "CAMEO", the slightly different classifications could allow for a more refined cluster analysis. So I decided to keep both instead of dropping one.
# 
# For other mixed-type features ('LP_LEBENSPHASE_FEIN','LP_LEBENSPHASE_GROB') were dropped on the basis that the information in these features were already included in other features in the dataset, 1.2 "ALTERSKATEGORIE_GROB", 1.13 "LP_FAMILIE_FEIN" and 1.15 "LP_STATUS_FEIN" and it wasn't necessary to re-engineer and keep them in the dataset.
# 
# For "KBA05_BAUMAX", again this feature was a summary of other features in the RR3 micro-cell category and was deemed unnecessary to re-engineer and keep.
# 
# However, for the last feature, 'WOHNLAGE', I thought the quality of the neighborhood could impact customer segments and this information wasn't available in other features, I re-engineered "WOHNLAGE" into 2 features, the first a binary categorical feature, whether the neighbourhood was rural or not, and the second into a rating class for the neighborhood. For rural neighborhoods I classed them as no score calculated.

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[89]:


# Review of features 
azdias_2[:1].transpose()


# In[90]:


# Drop column "MIN_GEBAEUDEJAHR" for irrelevance to customer segments
azdias_2 = azdias_2.drop('MIN_GEBAEUDEJAHR',axis=1)


# In[91]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)

azdias_2.to_csv('azdias_checkpoint_2.csv',sep=';',encoding='utf-8',index=False)


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[92]:


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv',sep=';')
    
    for i,V in enumerate(df.iteritems()):
        missing_unknown = feat_info['missing_or_unknown'][i]
        column_name = V[0]
        missing_unknown = missing_unknown[1:-1].split(',')
        if missing_unknown != ['']:
            hold = []
            for x in missing_unknown:
                if x in ['X','XX']:
                    hold.append(x)
                else:
                    hold.append(int(x))
            df[column_name] = df[column_name].replace(hold,np.nan)
    
    # remove selected columns and rows, ...
    df = df.drop(['TITEL_KZ','AGER_TYP','KK_KUNDENTYP','KBA05_BAUMAX','GEBURTSJAHR','ALTER_HH'],axis=1)
    missing_row_data = df.isnull().sum(axis=1)
    df = df[df.index.isin(missing_row_data[missing_row_data <= 9].index)]
    
    # select, re-encode, and engineer column values.
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].map({'W': 1, 'O': 0})
    df = df.drop(['LP_FAMILIE_GROB','LP_STATUS_GROB','GEBAEUDETYP','CAMEO_DEU_2015'],axis=1)
    multi = ['CJT_GESAMTTYP','FINANZTYP','GFK_URLAUBERTYP','LP_FAMILIE_FEIN','LP_STATUS_FEIN',
             'NATIONALITAET_KZ','SHOPPER_TYP','ZABEOTYP','CAMEO_DEUG_2015']
    df = pd.get_dummies(df, columns = multi, prefix=multi)
    
    avantgarde = []

    for x in df['PRAEGENDE_JUGENDJAHRE']:
        if x in [1,3,5,8,10,12,14]:
            avantgarde.append(0) # Value 0 denotes 'Mainstream'
        elif x in [2,4,6,7,9,11,13,15]:
            avantgarde.append(1) # Value 1 denotes 'Avantgarde'
        else:
            avantgarde.append(np.nan)
    
    a = pd.Series(avantgarde)
    df['AVANTGARDE'] = a.values
    
    df['DEKADE'] = df['PRAEGENDE_JUGENDJAHRE'].map({1:1,
                        2:1,3:2,4:2,5:3,6:3,7:3,8:4,9:4,10:5,11:5,12:5,13:5,14:6,15:6})
    
    df = df.drop('PRAEGENDE_JUGENDJAHRE',axis=1)
    
    cameo_wealth = []
    for x in df['CAMEO_INTL_2015']:
        x = float(x)
        if 10 < x < 16:
            cameo_wealth.append(5)
        elif 20 < x < 26:
            cameo_wealth.append(4)
        elif 30 < x < 36:
            cameo_wealth.append(3)
        elif 40 < x < 46:
            cameo_wealth.append(2)
        elif 50 < x < 56:
            cameo_wealth.append(1)
        else:
            cameo_wealth.append(np.nan)

    cw = pd.Series(cameo_wealth)
    df['CAMEO_WEALTH'] = cw.values
    
    cameo_lifestage = []
    for x in df['CAMEO_INTL_2015']:
        x = float(x)
        if x % 10 == 1:
            cameo_lifestage.append(1)
        elif x % 10 == 2:
            cameo_lifestage.append(2)
        elif x % 10 == 3:
            cameo_lifestage.append(3)
        elif x % 10 == 4:
            cameo_lifestage.append(4)
        elif x % 10 == 5:
            cameo_lifestage.append(5)
        else:
            cameo_lifestage.append(np.nan)
            
    cl = pd.Series(cameo_lifestage)
    df['CAMEO_LIFESTAGE'] = cl.values
    
    df['RURAL'] = df['WOHNLAGE'].map({0:0,1:0,2:0,3:0,4:0,5:0,7:1,8:1})
    df['GEGEND'] = df['WOHNLAGE'].map({0:0,1:1,2:2,3:3,4:4,5:5,7:0,8:0})
    
    df = df.drop(['CAMEO_INTL_2015','LP_LEBENSPHASE_FEIN','LP_LEBENSPHASE_GROB','WOHNLAGE','PLZ8_BAUMAX','MIN_GEBAEUDEJAHR'],
                 axis=1)
    # Return the cleaned dataframe.
    
    return df


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[93]:


# Load last checkpoint
azdias_3 = pd.read_csv('azdias_checkpoint_2.csv',sep=';')


# In[94]:


# Check shape
azdias_3.shape


# In[95]:


# Take a list of column headers
header_list = list(azdias_3.columns.values)
header_list


# In[96]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.

imputer = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
azdias_3 = imputer.fit_transform(azdias_3)
azdias_3 = pd.DataFrame(azdias_3)
azdias_3.head()


# In[97]:


# Apply feature scaling to the general population demographics data.
scaler = StandardScaler()
azdias_3 = scaler.fit_transform(azdias_3)
azdias_3 = pd.DataFrame(azdias_3,columns=header_list)
azdias_3.head()


# In[98]:


# Make a copy ready for Principal Component Analysis
azdias_pca_test = azdias_3.copy()
azdias_pca_test.head()


# ### Discussion 2.1: Apply Feature Scaling
# 
# As discussed in section 1.1.3, I have tried to minimize the quantity of values imputed in the dataset to maintain the truthfulness of the dataset distributions, and probably would have avoided doing any imputing in a real-life scenario for the reasons stated in step 1.1.3.
# 
# For the imputing I chose to use the mode or 'most frequent' value rather than the mean or median to fill in the missing values. I thought this was the best of the 3 options with the majority of the features being categorical or ordinal in nature and only 7 features being numerical.
# 
# The mean and median would have squashed the dataset distibutions towards a center which as no relevance with categorical and ordinal distributions. The mode seemed the logical choice as it would reinforce the highest value for each feature making them stand out more in the later analysis for customer segments.
# 
# Then I used the StandardScaler has suggested for the scaling.
# 

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[99]:


# Apply PCA to the data.
def do_pca(n_components,data):
    #X = StandardScaler().fit_transform(data)
    X = data
    pca = PCA(n_components,whiten=True,random_state=42)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


# In[100]:


pca, azdias_pca_test = do_pca(132,azdias_pca_test)


# In[101]:


azdias_pca_test = pd.DataFrame(azdias_pca_test)
azdias_pca_test.head()


# In[102]:


# Investigate the variance accounted for by each principal component.
def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components
    INPUT: pca - the result of instantian of PCA in scikit learn
    OUTPUT: none
    '''
    sns.set_style("darkgrid")
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    
    plt.figure(figsize=(15,9))
    ax = plt.subplot(111)
    plt.style.use("ggplot")
    
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals,color='teal')
    for i in range(0,ind[5]):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2,vals[i]), va='bottom',ha='center',fontsize=15)
        
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=1,length=12)
    
    ax.set_xlabel("Principal Component",fontsize=13.5)
    ax.set_ylabel("Variance Explained (%)",fontsize=13.5)
    plt.title("Explained Variance per Principal Component",fontsize=15)
    
    plt.show()
    
    
scree_plot(pca)


# In[103]:


def scree_plot2(pca):
    '''
    Creates a scree plot associated with the principal components
    INPUT: pca - the result of instantian of PCA in scikit learn
    OUTPUT: none
    '''
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    
    plt.figure(figsize=(15,9))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    
    ax.plot(ind, cumvals,color='teal',linewidth=3)
    plt.hlines(y=0.8, xmin=0, xmax=56, color='red', linestyles='dashed',zorder=1)
    plt.vlines(x=56, ymin=0, ymax=0.8, color='red', linestyles='dashed',zorder=2)
    plt.hlines(y=0.3, xmin=0, xmax=4, color='red', linestyles='dashed',zorder=3)
    plt.vlines(x=5, ymin=0, ymax=0.29, color='red', linestyles='dashed',zorder=4)
    plt.hlines(y=0.5, xmin=0, xmax=20, color='red', linestyles='-',zorder=5)
    plt.vlines(x=20, ymin=0, ymax=0.5, color='red', linestyles='-',zorder=6)
    
    ax.set_xlabel("Principal Component",fontsize=13.5)
    ax.set_ylabel("Variance Explained (%)",fontsize=13.5)
    plt.title("Explained Cumulative Variance per Principal Component",fontsize=15)
    
    print(cumvals)
    
scree_plot2(pca)


# In[104]:


# Calculate cumulative percentage of variance captured by number of principal components
pca1 = pca.explained_variance_ratio_.tolist()
print(np.sum(pca1[:21])) 


# In[113]:


# Re-apply PCA to the data while selecting for number of components to retain.
# 1st Choice based on trying to keep more than 80% of the dataset
pca_2, azdias_30 = do_pca(132,azdias_3)


# In[114]:


# 2nd Choice based on simplicity of segment results. The first 5 components make up 30% of the dataset variance
pca_3, azdias_31 = do_pca(132, azdias_30)


# In[115]:


# Final choice is halfway house, 21 principal components making up 50% of the dataset variance
pca_4, azdias_32 = do_pca(21, azdias_31)


# In[116]:


azdias_33 = pd.DataFrame(azdias_32)
azdias_33[:5]


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# 
# Before performing dimensionality reduction we have 132 features in the dataset, cleaned, imputed, scaled and ready to use. After running these 132 features through principal component analysis to obtain a component variance analysis and visualizing them on a scree plot. My initial thought was to lose as little of the dataset variance as possible, which reflected my approach through the data wrangling process, while reducing the number of principal components to be used in the cluster segment analysis. I have used (principal components) 132 to maintained more taht 80% of the dataset variance.
# 
# However, I was concerned I was oversimplifying, 30% seemed too low to obtain an accurate analysis! I researched on the internet but found contradictory approaches and opinions, but domain knowledge seems to be most relevant in making the decision of what level of variance loss is acceptable. I do not have the domain knowledge so decided to be a little more conservative in my approach and chose finally a middle-of-the-road approach, keeping 50% of the dataset variance with 21 Principal Components.

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[117]:


# Quick look at the composition of the components.
pca_4.components_


# In[118]:


# Quick look at the value of variance explained by each component
pca_4.explained_variance_ratio_


# In[119]:


# Create a dataframe of Explained Variance
dimensions = ['PC {}'.format(i) for i in range(1,len(pca_4.components_)+1)]
ratio = pd.DataFrame(pca_4.explained_variance_ratio_,columns = ['EXPLAINED_VARIANCE'])
ratio = ratio.round(4)
ratio.index = dimensions
ratio[:21]


# In[120]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.

weights = pd.DataFrame(pca_4.components_, columns=header_list)
weights = weights.round(4)
weights.index = dimensions
weights[:5]


# In[121]:


# Combine the 2 dataframes
result = pd.concat([ratio, weights], axis = 1, join_axes=[ratio.index])
result[:3]


# In[122]:


# Function integrating the steps above to be used in later analysis
def show_pca_weights(principal_component,no_of_weights):
    
    dimensions = ['PC {}'.format(i) for i in range(1,len(pca_4.components_)+1)]
    
    ratio = pd.DataFrame(pca_4.explained_variance_ratio_,columns = ['EXPLAINED_VARIANCE'])
    ratio = ratio.round(4)
    ratio.index = dimensions
    
    weights = pd.DataFrame(pca_4.components_, columns=header_list)
    weights = weights.round(4)
    weights.index = dimensions
    
    result = pd.concat([ratio, weights], axis = 1, join_axes=[ratio.index])
    result[:5]
    print("Principal Component", (principal_component))
    print('-' * 30)
    print(result.iloc[(principal_component)-1].sort_values(ascending=False)[:no_of_weights])
    print('-' * 30)
    print(result.iloc[(principal_component)-1].sort_values()[:no_of_weights])


# In[123]:


# Analysis of the highest positive and negative weights for the first Principal Componnent
show_pca_weights(1,5)


# In[124]:


# Analysis of the highest positive and negative weights for the second Principal Componnent
show_pca_weights(2,5)


# In[125]:


# Analysis of the highest positive and negative weights for the third Principal Componnent
show_pca_weights(3,5)


# In[126]:


# Save checkpoint
azdias_33.to_csv('azdias_checkpoint_3.csv',sep=';',encoding='utf-8',index=False)


# ### Discussion 2.3: Interpret Principal Components
# 
# Each component is a combination of the 132 features left after cleaning, with each feature given a weight relative to its importance for a particular principal component. The higher the weight either positive or negative, the more impact the feature has on the calculation of the principal component. What value of weight is important is subjective and depends on the context.
# 
# I have taken the 5 highest weights positive and negative to interpret the first 3 principal components, this is an arbitrary decision and I could easily have extended this to 6 or reduced it to only 4 as the weight ratios are close to each other in numeric value.
# 
# For component 1 the top 5 positive weights are: "PLZ8_ANTG3", number of 6-10 family homes in the PLZ8 region, "HH_EINKOMMEN_SCORE", Estimated household net income, "PLZ8_ANTG4", number of 10+ family houses in the PLZ8 region, "ORTSGR_KLS9", size of community, "EWDICHTE", density of households per km squared.
# 
# The top 5 negative weights are: "FINANZ_MINIMALIST", low financial interest, "MOBI_REGIO", movement patterns, "PLZ8_ANTG1", number of 1-2 family houses in PLZ8 region, "KBA05_ANTG1" number of 1-2 family houses in the microcell, "KBA05_GBZ", number of buildings in the microcell.
# 
# The first principal component seems to be representing a combination of features that relate to population density in an area and the associated wealth of that area. Positive correlations with increasing population density and wealth and negative correlations with low density and little financial interest features.
# 
# For component 2 the top 5 positive weights: "ALTERSKATEGORIE_GROB", estimated age based on given name analysis, "FINANZ_VORSORGER", financial typology is be prepared, "ZABEOTYP_3", energy consumption typology, "SEMIO_ERL", personality typology event-oriented, "RETOURTYP_BK_S", return shopper type.
# 
# And top 5 negative weights: "DEKADE", decade of youth, "FINANZ_SPARER", financial typology is money-saver, "SEMIO_REL", personality typology is religious, "FINANZ_UNAUFFAELLIGER", financial typology is inconspicuous, "SEMIO_TRADV", personality typology is traditional-minded.
# 
# The second principal component seems to be representing a combination of features related to individuals of older people with conservative traditional values. Positive correlations with increasing age and financial prudence.
# 
# For component 3, I have taken only the first 4 positive features as the fifth as a weight conspicuously lower than the fourth at 14% to 25%. The top weight "SEMIO_VERT", personality type dreamful, is also 6% higher than the second weight "SEMIO_FAM", personality type is family-minded, "SEMIO_SOZ", personality type is socially-minded, and "SEMIO_KULT", personality type is cultural-minded.
# 
# And the top 5 negative weights: "ANREDE_KZ" gender at -35%, "SEMIO_KAEM", personality typology is combative attitude, "SEMIO_DOM", personality typology is dominant-minded, "SEMIO_KRIT", personality typology is critical-minded, "SEMIO_ERL", personality typology is event-orientated.
# 
# Finally, the third principal component seems to be representing a combination of features related to individuals of female gender with a humanitarian disposition.

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[127]:


# Over a number of different cluster counts...

azdias_4 = pd.read_csv('azdias_checkpoint_3.csv',sep=';')
azdias_4[:1]
    # run k-means clustering on the data and...
def get_kmeans_score(data, center):
    
    kmeans = KMeans(n_clusters = center, n_init=10, max_iter=300, random_state=42)
    model = kmeans.fit(data)
    score = np.abs(model.score(data))
    return score

score = []
#centroids = [2,4,6,7,8,9,10,11,12,13,14,15,16,18,20,22,25]
centroids = [1,3,6,7,9,12,15,16,18,21,25]

for x in centroids:
    score.append(get_kmeans_score(azdias_4,x))
    print("Current Centroid:",x)

    # compute the average within-cluster distances.
score_ds = pd.Series(score)
centroids_ds = pd.Series(centroids)
score_df = pd.concat([centroids_ds,score_ds],axis=1)
score_df.columns = ['clusters','average_distance']
score_df.set_index('clusters',inplace=True)
score_df['average_distance'] = score_df['average_distance'].round(0)
score_df[:17] 
    


# In[128]:


# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.

fig = plt.subplots(figsize=(16,10))
ax = plt.subplot(111)
plt.plot(centroids,score,linestyle='--',linewidth=3,marker='o',color='teal')

ax.set_xlabel('K-centroids', fontsize=13.5)
ax.set_ylabel('Score - Sum of Squared Errors', fontsize=13.5)
plt.title('Sum of Squared Errors v Number of Centroids',fontsize=15);


# In[129]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.

kmeans_10 = KMeans(n_clusters=10, n_init=10, max_iter=300, random_state=42)
population_clusters = kmeans_10.fit_predict(azdias_4)


# ### Discussion 3.1: Apply Clustering to General Population
# 
# The first run of K-Means with 132 principal components resulted in almost a straight line with little definition of an "elbow". 
# 
# However, my final choice of 21 principal components resulted in the plot and table you see above. There is an "elbow", but it is much less defined than with 5 principal components. Possible cluster candidates are 6, 7, 10, 12, and 15 depending if you take a high "elbow" at 6 clusters or low "elbow" at 15 clusters. Again from research on the internet, there doesn't seem to be an agreement on a "correct" location to decide the number of clusters, knowledge of the domain is useful. I also noticed that changing the plotsize and changing the height and width dimensions of the plot affected the perception of where was the best "elbow".
# 
# I ran all the clusters mentioned above through the population and customer datasets to be able to make comparisons on the effects of selecting a different number of clusters. In general, the lower number of clusters gave more equal distribution in the population clustering of the dataset, there was less extremes between the clusters, whereas a higher number of clusters resulted in the population clustering having some wide differences between counts allocated to clusters. The customer distribution varied according to the number of clusters.
# 
# Once again, due to my inexperience with these techniques, I went for a 'middle' solution and chose 10 clusters which seems to reflect the mid-point in the bend of the "elbow".

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[130]:


# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv',sep=';')


# In[131]:


# View shape of customer dataset
customers.shape


# In[132]:


# Quick check of customer dataframe
customers.head()


# In[133]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.

customers_clean = clean_data(customers)

customers_clean.shape

customers_clean[:10]


# In[134]:


customer_header_list = list(customers_clean.columns.values)
customer_header_list


# In[135]:


# Check for null values
customers_clean.isnull().any().sum()


# In[136]:


# Use population imputer to transform customer dataframe
customers_clean = imputer.transform(customers_clean)
customers_clean = pd.DataFrame(customers_clean)
customers_clean.head()


# In[137]:


# Check no null values remain
customers_clean.isnull().any().sum()


# In[138]:


# Apply population scaler to transform customer dataframe
customers_clean = scaler.transform(customers_clean)
customers_clean = pd.DataFrame(customers_clean)
customers_clean.head()


# In[139]:


# Apply population Principal Component transformation on customer dataframe
customers_clean = pca_4.transform(customers_clean)
customers_clean = pd.DataFrame(customers_clean)
customers_clean.head()


# In[140]:


# Apply K-means clustering prediction using fitted algorithm from population
customer_clusters = kmeans_10.predict(customers_clean)


# In[141]:


# Obtain missing row data for customer dataset
missing_row_data_2 = customers.isnull().sum(axis=1)
customer_rows_deleted = customers[customers.index.isin(missing_row_data_2[missing_row_data_2 > 9].index)]
customer_rows_deleted[:4]


# In[142]:


customer_rows_deleted.shape


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[143]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.
customer_clusters = pd.Series(customer_clusters)
cc = customer_clusters.value_counts().sort_index()
cc = pd.Series(cc)
cc


# In[144]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?
population_clusters = pd.Series(population_clusters)
pc = population_clusters.value_counts().sort_index()
pc = pd.Series(pc)
pc


# In[145]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?
final_df = pd.concat([pc, cc], axis=1).reset_index()
final_df.columns = ['cluster','population','customer']
final_df


# In[146]:


# Visualize dataframe as bar charts
fig = plt.figure(figsize=(16,6))

ax1 = fig.add_subplot(1,2,1)
ax1 = sns.barplot(x='cluster',y='population',color='teal',edgecolor='.9',data=final_df)
plt.title("Distribution of General Population into Clusters")

ax2 = fig.add_subplot(1,2,2)
ax2 = sns.barplot(x='cluster',y='customer',color='coral',edgecolor='.9',data=final_df)
plt.title("Distribution of Customers into Population Clusters")
plt.suptitle("Plot 3.3.1", fontsize=15);


# In[147]:


# Add ratio and ratio difference for each cluster to the dataframe
final_df['pop_%'] = (final_df['population']/final_df['population'].sum()*100).round(2)
final_df['cust_%'] = (final_df['customer']/final_df['customer'].sum()*100).round(2)
final_df['diff'] = final_df['cust_%'] - final_df['pop_%']
final_df


# In[148]:


# Visualize ratios for each cluster

fig = plt.figure(figsize=(16,6))

ax = fig.add_subplot(111)

ax = final_df['pop_%'].plot(x=final_df['cluster'], kind='bar',color='teal',width=-0.3, align='edge',position=0)
ax = final_df['cust_%'].plot(kind='bar',color='coral',width = 0.3, align='edge',position=1)

ax.margins(x=0.5,y=0.1)
ax.set_xlabel('Clusters', fontsize=15) 
ax.set_ylabel('Ratio %', fontsize=15)
ax.xaxis.set(ticklabels=[0,1,2,3,4,5,6,7,8,9,10,11])
ax.tick_params(axis = 'x', which = 'major', labelsize = 16)
plt.xticks(rotation=360,)

plt.legend(('poulation %', 'customer %'),fontsize=15)
plt.title(('Comparison of ratio of general population and customer segments as % of total in each cluster.')
          ,fontsize=16)

plt.subplots_adjust(bottom=0.2)
plt.suptitle("Plot 3.3.2", fontsize=15)
plt.show()


# In[149]:


# Missing rows from population dataset
population_rows_deleted = azdias_subset_2.shape[0]
population_rows_deleted


# In[150]:


# Analysis of principal components of cluster 7 with over-representation in customer segment.
cc0 = kmeans_10.cluster_centers_[7]
cc0 = pd.Series(cc0)
cc0.index = cc0.index +1
cc0.sort_values(ascending=False)


# In[151]:


# Transform cluster 7 to original feature values.
cc00 = scaler.inverse_transform(pca_4.inverse_transform(cc0))
cc00 = pd.Series(cc00).round(2)
cc00.index = header_list
cc00


# In[152]:


# Analyze top principal component in cluster 7
show_pca_weights(4,5)


# In[153]:


# Analyze second top principal component in cluster 7
show_pca_weights(11,5)


# In[154]:


# Analysis of principal components of cluster 6 with under-representation in customer segment.
customer_under_rep = kmeans_10.cluster_centers_[6]
customer_under_rep = pd.Series(customer_under_rep)
customer_under_rep.index = customer_under_rep.index +1
customer_under_rep.sort_values(ascending=False)


# In[155]:


# Transform cluster 7 to original feature values.
customer_under_rep_features = scaler.inverse_transform(pca_4.inverse_transform(customer_under_rep))
customer_under_rep_features = pd.Series(customer_under_rep_features).round(2)
customer_under_rep_features.index = header_list
customer_under_rep_features


# In[156]:


# Analyze top principal component in cluster 6
show_pca_weights(1,5)


# In[157]:


# Analyze second top principal component in cluster 6
show_pca_weights(7,5)


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# First of all, if we compare the Population and Customer clusters without consideration of the missing row data, we can see that the general population dataset:
# Cluster 1 has the largest count of 1.5m
# Cluster 9 has the lowest count with just 3241.
# Whereas for the customer dataset:
# Cluster 7 has the largest count of 56K
# Cluster 5 has the lowest count of 892 observations.
# 
# From plot 3.3.1, you can see there is a difference in the cluster distributions between the general population and the customer dataset. Looking at the ratios of each cluster against total counts in order to make better comparisons, we can see the customer dataset is over-represented in clusters 7 and 0 at greater than 5% difference to the general population, and under-represented in clusters 6, 2, 3, and 8 at the 5% difference level.
# 
# This can be seen in Plot 3.3.2, where the significance of cluster 7 customer segment to the mail order company's customer base is clearly visible.
# 
# However, to make this analysis more complete, we should also include the missing row data as an additional cluster. In step 1.1.3 we calculated that the rows of missing data were qualitatively different to the the rows kept for the analysis.
# 
# Plots 3.3.3 and 3.3.4 are the same visualizations as 3.3.1 and 3.3.2 respectively with the missing row data added as cluster -1. We can see the impact is significant. The missing row data for the population represented 13% of the total but made up 31% of the customer data with a count of 61,179 observations. This is higher than our largest cluster number 7 with a count of 56,031.
# 
# So, our ratio differences at the 5% threshold, the customer dataset is over-represented in clusters -1 and 7, and under-represented in cluster 6, clusters 2, 8, 3 and 1 are also just above the 5% difference.
# 
# The missing row data requires more investigation and analysis but from the preliminary bar charts constructed in step 1.1.3, this data seems to represent conservative, minimilistic high-income earners receptive to advertising incentives and is a relevant customer segment for targeting for the mail order company.
# 
# Cluster 7 is the most relevant cluster for the mail order company's customer base. If we analyze the features that make up this cluster by inverse transforming the principal components and scaling we obtain values for the cluster centroid for our original features after data wrangling was completed that we can use to create a picture of the customer segment. We can also determine the most important features by looking at the principal components that have the biggest impact on the cluster and then seeing which features make up the biggest positive and negative weights of this principal component as we did in step 2.3.
# 
# For cluster 7, the main principal component is number 4 that has a value of 1.35 followed by component number 11, 17 and 3 with values of 0.6, 0.39 and 0.35 respectively. From this we can see principal component 4 is by far the biggest influence.
# 
# Looking at the top five positive weights for component 4 and their original feature value from the inverse transformation:
# 1."AVANTGARDE", feature value = 0.98, signifies avantgarde individual
# 2."GREEN_AVANTGARDE", feature value = 0.98, signifies very high green environmental values
# 3."LP_STATUS_FEIN_10.0", feature value = 0.76, signifies top earner
# 4."EWDICHTE", feature value = 3.91, signifies medium density population area
# 5."ORTSGR_KLS9", feature value = 5.31, signifies community of 20,000-50,000 inhabitants
# 
# Top five negative weights for component 4 and their original feature value:
# 1."RURAL", feature value = 0.15, signifies not in a rural area
# 2."HH_EINKOMEN_SCORE", feature value = 2.30, signifies individual as very high income
# 3."BALLRAUM:, feature score = 4.05, signifies lives 30-40km from urban center
# 4."INNENSTADT", feature score = 4.63, signifies lives 5-10km from city center
# 5."LP_STATUS_FEIN_4.0", feature score = -0.01, signifies not a village
# 
# I also looked at the significant weights in principal component 11. Top 3 positive weights are:
# 1."FINANZTYP_6", feature score = 0.43, signifies average conspicuous spender
# 2."PLZ8_HHZ", feature score = 3.69, signifies about 400 households in PLZ8 region
# 3."KBA13_ANZAHL_RKW", feature score = 701, signifies number of cars in PLZ8 region
# 
# Top 3 negative weights for principal component 11:
# 1."OST_WEST_KZ", feature score = 0.89, individuals mainly from old West Germany
# 2."FINANZTYPE_5", feature score = 0.18, individual not an investor
# 3."FINANZTYP_UNAUFFAELLIGER", feature score = 2.28, average to high conspicuous spender
# 
# From the above details we can start to form a picture of the biggest customer segment group for the mail order company. They are high income earners with disposable income and a very strong affinity to green/environmental issues that keep abreast with the latest technological developments. They live in the old Western part of Germany in smallish urban areas but within driving distance of a larger urban city.
# 
# Cluster 6 is the least relevant customer segment for the company. The main principal components in this cluster are principal component 1 with a value of 1.18 and principal component 7 with a value of 0.35, conspicuously less. If we follow the same process of analysis as above to form a picture of this group underrepresented.
# 
# Looking at the top five positive weights for component 1 and their original feature value from the inverse transformation:
# 1."PLZ8_ANTG3", feature value = 2.22, signifies there is an average to high number of 6-10 family houses
# 2."HH_EINKOMEN_SCORE", feature value = 5.47, signifies lower income to very low income individuals
# 3."PLZ8_ANTG4", feature value = 1.07, signifies there is a low number of 10+ family homes
# 4."ORTSGR_KLS9", feature value = 6.23, signifies community of 50,000-100,000 inhabitants
# 5."EWDICHTE", feature value = 4.65, signifies medium to high density of households
# 
# Top five negative weights for component 1 and their original feature value:
# 1."FINANZ_MINIMALIST", feature value = 1.19, signifies individual with very high low financial interest
# 2."MOBI_REGIO", feature value = 2.07, signifies high movement individuals
# 3."PLZ8_ANTG1", feature score = 1.68, signifies low/average single/couple homes in macrocell PLZ8
# 4."KBA05_ANTG1", feature score = 0.65, signifies low share of single/couple homes in microcell RR4
# 5."KBA05_GBZ", feature score = 2.48, signifies number of buildings in microcell is between 3 and 16
# 
# Top 3 positive weights in principal component 7:
# 1."FINANZTYP_HAUSBAUER", feature score = 4.26, signifies low home ownership
# 2."SHOPPER_TYP_3.0", feature score = 0.14, signifies not a discerning shopper
# 3."KBA05_ANTG4", feature score = 0.41, signifies there is a low quantity of 10+ family houses in microcell
# 
# Top 3 negative weights for principal component 7:
# 1."OST_WEST_KZ", feature score = 0.77, more individuals from old West Germany
# 2."PLZ8_GBZ", feature score = 2.97, signifies there are 130-299 buildings in macrocell PLZ8
# 3."KBA13_ANZAHL_PHK", feature score = 568, lower number of cars in macrocell
# 
# The picture we start to see of the individuals who are not customers for the mail order company. They are low to very low income earners that are living in moderately dense urban towns and small cities with buildings that contain many apartments. They don't own these apartments and are probably renting. They have no interest in financial matters and move around alot as well as being not discernible shoppers. The above description makes me think this group could be students but I would also have expected an age feature to be present or possible migrant workers on minimum pay.
# 
# My final thoughts and takeaways from this project, on reflection I think I was too careful with the dataset at the data wrangling stage. Looking back I think I could have been more ruthless in removing columns, like the vacation analysis 1.8 "GFK_URLAUBERTYP", like many of the columns give equivalent information, and I could have deleted more rows to obtain a more concise dataset that was less detailed. I think a better approach is to obtain a customer segment summary analysis using generalized population data, no imputing, few principal components and finally few clusters. Once this have been done, and some general clusters identified. You can then return to the more detailed column information and redo the general clusters into more detailed sub-cluster breakdowns.
# 
# For the record, this project was very interesting and enjoyable to work through.

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:




