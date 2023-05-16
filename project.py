# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# load in the data
azdias = pd.read_csv('./Data/Udacity_AZDIAS_052018.csv', \
                     sep=';',low_memory=False)
customers = pd.read_csv('./Data/Udacity_CUSTOMERS_052018.csv', \
                        sep=';', low_memory=False)

# First we try to wrangle azdias and prepare it to start our analysis
# calculating null percentage of each column to decide which column to be eliminated
for i in range(azdias.shape[1]):
    print('azdias: Null percentage for column {} is {:.1f}%'.\
          format(i,(azdias.iloc[:,i].isnull().sum()/azdias.shape[0])*100))
    

# We see that column 300 is relevant to cutomer status which is important for our 
# modelling
# while 65.6% of this column is null. We then decide to change all nan values to 
# -1 which means unknown
azdias.iloc[:,300].fillna(-1,inplace=True)

# Column 100 is relevant to EXTSEL992 key that has no definition in the excel 
# file with 73.4% null vlues
# we then think that this column is not important for the modelling and drop 
# whole the column
azdias.drop(['EXTSEL992'],axis=1,inplace=True)


# Next step is to drop all columns with more than 90 percent of null data as 
# imputing would not reflect the precise action for these columns. 
for i in range(azdias.shape[1]):
    if(azdias.iloc[:,i].isnull().sum()/azdias.shape[0]*100>90):
        print('azdias: Null percentage more than 90%\
              for column {} with null percentage {:.1f}%'.\
                  format(i,(azdias.iloc[:,i].isnull().sum()\
                            /azdias.shape[0])*100))


# we then decide to eliminate each column with more than 90% null data
dropped_cols = azdias.columns[4:8]
azdias.drop(dropped_cols,axis=1,inplace=True)


# Check if there is one record with more than 90% null data. We proceed with 
# dropping this record and continue with imputing null records for all the columns
for i in range(azdias.shape[0]):
    if((azdias.iloc[i,:].isnull().sum()/azdias.shape[1])*100>90):
        print('azdias: Record {} has more than 90% null columns with null\
              percent {:.1f}%'.\
                  format(i,(azdias.iloc[i,:].isnull().sum()/azdias.shape[1])*100))

# Result is no record having more than 90% of null data. We thus continue with 
# imputing null records for all the remaining columns. We choose mode of each 
# column to replace null values as we are working with categorical data for all 
# the columns.
for col in azdias.columns:
    try:
        azdias[col] = azdias[col].transform(lambda x: x.fillna(x.mode()[0]))
    except:
        print('That broke...')


# Checking null values
for i in range(azdias.shape[1]):
    print('azdias: Null percentage for column {} is {:.1f}%'.\
          format(i,(azdias.iloc[:,i].isnull().sum()/azdias.shape[0])*100))

# We do the same data wrangling for customers dataframe
# First we try to wrangle customers df and prepare it to start our analysis
# calculating null percentage of each column to decide which column to be eliminated
for i in range(customers.shape[1]):
    print('customers: Null percentage for column {} is {:.1f}%'.\
          format(i,(customers.iloc[:,i].isnull().sum()/customers.shape[0])*100))


# we decide to eliminate each column with more than 90% null data
dropped_cols = customers.columns[4:8]
customers.drop(dropped_cols,axis=1,inplace=True)

# for all the remaining columns. We choose mode of each column to replace null values as we are working 
# with categorical data for all the columns.
for col in customers.columns:
    try:
        customers[col] = customers[col].transform(lambda x: x.fillna(x.mode()[0]))
    except:
        print('That broke...')


# Checking null values
for i in range(customers.shape[1]):
    print('customers: Null percentage for column {} is {:.1f}%'.\
          format(i,(customers.iloc[:,i].isnull().sum()/customers.shape[0])*100))


# ## Part 1: Customer Segmentation Report

# This is the algorithm: we first prepare datasets as they have new dummy variables. 
# Then dimensional reduction will be done for two datasets, considering elbow 
# method to find the optimum number of features to take up 90% of the total 
# variance of the original data. Then two datasets will be standardized and 
# clustered (again considering elbow method to find optimum number of clusters) 
# using K-means algorithm. The cluster labels then will be compared to find the 
# similarity between clusters employing some specific metrics. The final outcome 
# would be the clusters in two datasets that are mostly similar and the members 
# in two different datasets can represent eachother.


# Step 1
# Add dummy variables to customers and azdias datasets
print('Adding dummy variables to customers dataset ...')
customers = pd.get_dummies(customers)
print('Adding dummy variables to azdias dataset ...')
azdias = pd.get_dummies(azdias)

# standardize both datasets
azdias_scaler = StandardScaler()
customers_scaler = StandardScaler()
print('Standardizing azdias dataset ...')
azdias.iloc[:,1:] = azdias_scaler.fit_transform(azdias.iloc[:,1:])
print('Standardizing customers dataset ...')
customers.iloc[:,1:] = customers_scaler.fit_transform(customers.iloc[:,1:])

# writing the standardized datasets to hard drive
print('Writing standardized datasets ...')
azdias.to_csv('./Data/azdias_scaled.csv',index=False)
customers.to_csv('./Data/customers_scaled.csv',index=False)



# # In[16]:


# # Step 2
# # Now we proceed with dimensional reduction to a COMMON number of features that can fill up about 90% of the original data variance
# # We have first standardized both datasets
# # We use Dask to work with both datasets and also chunk data in each partition to reduce needed memory


# # In[17]:


# # Standardize the selected columns using Dask
# def standardize_chunk(chunk):
#     scaler = StandardScaler()
#     chunk[columns_to_standardize] = scaler.fit_transform(chunk[columns_to_standardize])
#     return chunk

# # Read customers dataframe to a Dask dataframe with 4 partitions
# dask_customers = dd.from_pandas(customers, npartitions=4)

# # Define the chunk size for processing for both datasets
# chunk_size = 10000  

# # Define the columns to be standardized (except the first column)
# columns_to_standardize = dask_customers.columns[1:]

# # Apply standardization on Dask DataFrame in chunks
# dask_customers = dask_customers.map_partitions(lambda df: df.groupby(df.index // chunk_size).apply(standardize_chunk))

# # Compute the result
# dask_customers = dask_customers.compute()


# # In[ ]:


# # Read azdias dataframe to a Dask dataframe with 16 partitions
# dask_azdias = dd.from_pandas(customers, npartitions=16)

# # Define the columns to be standardized (except the first column)
# columns_to_standardize = dask_azdias.columns[1:]

# # Apply standardization on Dask DataFrame in chunks
# dask_azdias = dask_azdias.map_partitions(lambda df: df.groupby(df.index // chunk_size).apply(standardize_chunk))

# # Compute the result
# dask_azdias = dask_azdias.compute()


# # In[ ]:


# pca=PCA()
# pca.fit(customers_scaled)


# # ## Part 2: Supervised Learning Model
# # 
# # Now that you've found which parts of the population are more likely to be customers of the mail-order company, it's time to build a prediction model. Each of the rows in the "MAILOUT" data files represents an individual that was targeted for a mailout campaign. Ideally, we should be able to use the demographic information from each individual to decide whether or not it will be worth it to include that person in the campaign.
# # 
# # The "MAILOUT" data has been split into two approximately equal parts, each with almost 43 000 data rows. In this part, you can verify your model with the "TRAIN" partition, which includes a column, "RESPONSE", that states whether or not a person became a customer of the company following the campaign. In the next part, you'll need to create predictions on the "TEST" partition, where the "RESPONSE" column has been withheld.

# # In[18]:


# mailout_train = pd.read_csv('./Data/Udacity_MAILOUT_052018_TRAIN.csv', sep=';')
# mailout_test = pd.read_csv('./Data/Udacity_MAILOUT_052018_TEST.csv', sep=';')


# # In[ ]:





# # In[ ]:




