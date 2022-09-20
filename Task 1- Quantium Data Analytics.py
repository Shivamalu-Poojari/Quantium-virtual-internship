# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 13:00:47 2022

@author: Shivamalu Poojari
"""
import pandas as pd
import seaborn as sns
import os

os.getcwd()
print(os.listdir())

os.chdir(r"C:\Users\Shashikala\OneDrive\Desktop\21\New folder")
dataset=pd.read_excel('QVI_transaction_data.xlsx')

dataset.shape
dataset.isnull().sum()
dataset.columns
# =============================================================================
# Summerization 
# =============================================================================
dataset.describe()

# =============================================================================
# Checking for outliers
# =============================================================================
sns.boxplot(dataset.TOT_SALES)
sns.displot(dataset.TOT_SALES, kde = True)

numericdata = dataset.select_dtypes(['float','int'])
numericdata.head()

# =============================================================================
# Removing outliers
# =============================================================================
x= numericdata[numericdata['TOT_SALES']< 8.000]

sns.displot(x.TOT_SALES, kde = True)
sns.boxplot(x.TOT_SALES)

# =============================================================================
# Data formats checking
# =============================================================================
dataset.dtypes