#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import chi2_contingency
from collections import Counter
from math import sqrt


# In[2]:


df = pd.read_csv(r'C:\Users\sofi\Downloads\wetransfer-3ab4ca\2018.csv',sep=';', encoding = 'ISO-8859-1',
                     names=['ID_ORDER', 'DATE-ADD','TIME-ADD','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE'],
                     dtype={'ID_ORDER':int,'DATE-ADD':object,'TIME-ADD':object,'LOCATION':int,'ID_CUSTOMER':object,'LAST_NAME':object,'FIRST_NAME':object,'YX_LIBELLE':object,'BIRTH_YEAR':object,'TELEX':object,'EMAIL':object,'ADRESS':object,'POSTAL_CODE':object,'CITY':object,'ITEM_CODE':object,'CC_LIBELLE':object,'CC_LIBELLE_1':object})

df


# In[3]:


df.info()


# In[4]:


df.replace({'0000-00-00': np.nan},inplace=True)


# In[5]:


df.isnull().sum()


# In[6]:


#convert DataFrame column to date-time:'GP_DATEPIECE'
df['DATE-ADD'] = pd.to_datetime(df['DATE-ADD'], format = '%Y-%m-%d')
df['DATE-ADD']= pd.to_datetime(df['DATE-ADD'], errors='ignore')
df.head()


# In[7]:


df['TIME-ADD'] = pd.to_datetime(df['TIME-ADD'])

df


# In[8]:


# Create the dictionary for  ETABLISSEMENT
etab = {28:'Sasio Geant',31:'Blue Island Carrefour',62:'Central Park',16:'Sousse',8:'Lafayette',130:'Blue Island Palmarium',50:'Sasio Carrefour',24:'Bizerte',40:'Ennasr',56:'Sasio Manzah VI',14:'Blue Island Zephyr',61:'La Soukra',63:'Sasio Menzah V',54:'Nabeul',37:'Sasio Zephyr',134:'Sasio Palmarium',64:'Nabeul',65:'Blue Island Manar',36:'Sasio Manar',25:'Blue Island Djerba',52:'Blue Island Menzah VI',66:'Mehdia',42:'Lac 2',67:'Sfax',68:'Monastir',51:'Blue Island Menzah V',69:'El Kef',35:'Kairouan',15:'Sasio Mseken',27:'Sasio Mseken',60:'Sasio Djerba',18:'Kelibia',41:'Ksar Hellal',74:'Hammamet'}
df['LOCATION'] = df['LOCATION'].map(etab)
#display the first 5 lines
df


# In[9]:


df['DATE-ADD'] = pd.to_datetime(df['DATE-ADD'])

df['SEASON'] = (df['DATE-ADD'].dt.month - 1) // 3
df['SEASON'] += (df['DATE-ADD'].dt.month == 3)&(df['DATE-ADD'].dt.day>=20)
df['SEASON'] += (df['DATE-ADD'].dt.month == 6)&(df['DATE-ADD'].dt.day>=21)
df['SEASON'] += (df['DATE-ADD'].dt.month == 9)&(df['DATE-ADD'].dt.day>=23)
df['SEASON'] -= 3*((df['DATE-ADD'].dt.month == 12)&(df['DATE-ADD'].dt.day>=21)).astype(int)


# In[10]:


season={0:'Winter',1:'Spring',2:'Summer',3:'Autumn'}

df['SEASON'] = df['SEASON'].map(season)
df


# In[11]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]
df


# In[12]:


df['COVID']='Pre-Covid'
df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','COVID','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]


# In[13]:


df['BIRTH_YEAR'] = pd.to_datetime(df['BIRTH_YEAR'],errors='coerce')
df.isnull().sum()


# In[14]:


df.BIRTH_YEAR[df.BIRTH_YEAR.dt.year > 2018] = np.nan
df.BIRTH_YEAR[df.BIRTH_YEAR.dt.year < 1918] = np.nan
df.isnull().sum()


# In[15]:


df['BIRTH_YEAR'].fillna((df['BIRTH_YEAR'].mean()), inplace=True)


# In[16]:


df.isnull().sum()


# In[17]:


df['BIRTH_YEAR'] = pd.to_datetime(df['BIRTH_YEAR'],errors='coerce')


# In[18]:


df['AGE'] = df['DATE-ADD'].dt.year- df['BIRTH_YEAR'].dt.year


# In[19]:


df['AGE'].max()


# In[20]:


df['AGE'].min()


# In[21]:


max_AGE=df['AGE'].max()
min_AGE=df['AGE'].min()


# In[22]:


cut_age = ['-25 ans','25-40ans', '40-65ans','+65 ans']
cut_bins =[min_AGE, 25, 40, 65, max_AGE]
df['AGE_SEGMENT'] = pd.cut(df['AGE'], bins=cut_bins, labels = cut_age)


# In[23]:


df['AGE_SEGMENT'].value_counts()


# In[24]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','COVID','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','AGE_SEGMENT','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]


# In[25]:


df['CONFINEMENT']='NO'
df['CURFEW']='NO'
df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','COVID','CONFINEMENT','CURFEW','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]
df


# In[26]:


df['P_MONTH']=0
df['P_MONTH']+=(df['DATE-ADD'].dt.day>10)
df['P_MONTH']+=(df['DATE-ADD'].dt.day>20)
df['P_MONTH']=df['P_MONTH'].astype(int)


# In[27]:


month={0:'Start_of_Month',1:'Middle_of_Month',2:'End_of_Month'}
df['P_MONTH'] = df['P_MONTH'].map(month)


# In[28]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','P_MONTH','COVID','CONFINEMENT','CURFEW','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]
df


# In[29]:


from datetime import date
import calendar


df['P_WEEK']=df['DATE-ADD'].dt.strftime('%A')
week={'Monday':'Start_of_Week','Tuesday':'Start_of_Week','Wednesday':'Mid_Week','Thursday':'Mid_Week','Friday':'Mid_Week','Saturday':'Week_End','Sunday':'Week_end'}
df['P_WEEK'] = df['P_WEEK'].map(week)


# In[30]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','P_MONTH','P_WEEK','COVID','CONFINEMENT','CURFEW','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]


# In[31]:


df['TOTAL_QUANTITY'] = df.groupby(['ID_ORDER'])['DATE-ADD'].transform('count')
df


# In[32]:


df2= pd.read_csv(r'C:\Users\sofi\Downloads\train_code_postal.csv')
df2 = df2.rename(columns={'code postal': 'POST_CODE'})
df2.info()


# In[33]:


hour=df['TIME-ADD'].dt.hour
hour


# In[34]:


# Create the dictionary for Hour 

H = {0:'Early_morning', 1:'Early_morning', 2:'Early_morning', 3:'Early_morning',4:'Early_morning',5:'Early_morning',6:'Early_morning',7:'Early_morning',8:'Early_morning',9:'Late_morning',10:'Late_morning',11:'Late_morning',12:'Early_afternoon',13:'Early_afternoon',14:'Early_afternoon',15:'Late_afternoon',16:'Late_afternoon',17:'Late_afternoon',18:'Evening',19:'Evening',20:'Evening',21:'Night',22:'Night',23:'Night'}
# Use the dictionary to map the 'Hour'
df['HOUR'] = hour.map(H)
#display the first 5 lines
df


# In[35]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','HOUR','SEASON','P_MONTH','P_WEEK','COVID','CONFINEMENT','CURFEW','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE','TOTAL_QUANTITY']]
df


# In[36]:


df['TIME-ADD'] = pd.Series([val.time() for val in df['TIME-ADD']])
df


# In[37]:


new_row2070 = {'POST_CODE':'2070', 'Delegation':'LA MARSA', 'poverty rate':2.2, 'zone':'urbaine','orientation':'nord-est'}
df2 = df2.append(new_row2070, ignore_index=True)


# In[38]:


df['POSTAL_CODE']=df['POSTAL_CODE'].apply(str)
df2['POST_CODE']=df2['POST_CODE'].apply(str)
df2.info()


# In[39]:


def get_region(code, train):
  try:
    if str(code) in str(train['POST_CODE'].unique()):
        return str(train[train['POST_CODE']== str(code)]['orientation'].values[0])
  except Exception :
    print (str(code))


# In[40]:


df['region'] = df.apply(lambda x: get_region(x.POSTAL_CODE, df2), axis=1)


# In[41]:


df['region'].value_counts()


# In[42]:


df['POSTAL_CODE'][df['region'].isnull()]


# In[43]:


def get_area(code, train):
  try:
    if str(code) in str(train['POST_CODE'].unique()):
        return str(train[train['POST_CODE']== str(code)]['zone'].values[0])
  except Exception :
    print (str(code))


# In[44]:


df['area'] = df.apply(lambda x: get_area(x.POSTAL_CODE, df2), axis=1)


# In[45]:


df = df.rename(columns={'region': 'REGION'})
df = df.rename(columns={'area': 'AREA'})


# In[46]:


cut_class = ['A', 'B', 'C+','C-', 'D', 'E']
cut_bins =[0, 2, 8, 18, 26, 42, 55]
df2['class'] = pd.cut(df2['poverty rate'], bins=cut_bins, labels = cut_class)
df2


# In[47]:


DF=pd.DataFrame()
DF['LOC']=df['LOCATION']
DF


# In[48]:


# Create the dictionary for  ETABLISSEMENT
POS = {'Sasio Geant':2032,'Blue Island Carrefour':2046,'Central Park':1000,'Sousse':4000,'Lafayette':1002,'Blue Island Palmarium':1000,'Sasio Carrefour':2046,'Bizerte':7000,'Ennasr':2083,'Sasio Manzah VI':2091,'Blue Island Zephyr':2070,'La Soukra':2035,'Sasio Menzah V':2091,'Nabeul':8000,'Sasio Zephyr':2070,'Sasio Palmarium':1000,'Nabeul':8000,'Blue Island Manar':2092,'Sasio Manar':2092,'Blue Island Djerba':4180,'Blue Island Menzah VI':2091,'Mehdia':5100,'Lac 2':1053,'Sfax':3100,'Monastir':5000,'Blue Island Menzah V':2091,'El Kef':7100,'Kairouan':3100,'Sasio Mseken':4070,'Sasio Mseken':4070,'Sasio Djerba':4180,'Kelibia':8090,'Ksar Hellal':5070,'Hammamet':8050}
DF['CODE'] = DF['LOC'].map(POS)
#display the first 5 lines
DF


# In[49]:


def get_class22(code, train):
  try:
    if str(code) in str(train['POST_CODE'].unique()):
      return str(train[train['POST_CODE']== str(code)]['class'].values[0])
  except Exception :
    print (str(code))


# In[50]:


DF['CLASS'] = DF.apply(lambda x: get_class22(x.CODE, df2), axis=1)


# In[51]:


DF = DF.rename(columns={'CLASS': 'class2'})


# In[78]:


df2[df2.POST_CODE=='7000']


# In[53]:


def get_class(code,code2, train,train2):
  try:
    if str(code) in str(train['POST_CODE'].unique()):
      return str(train[train['POST_CODE']== str(code)]['class'].values[0])
    else : 
        return str(train2[train2['LOC']== str(code2)]['class2'].values[0])
  except Exception :
    print (str(code))


# In[55]:


df['CLASS'] = df.apply(lambda x: get_class(x.POSTAL_CODE,x.LOCATION,df2,DF), axis=1)


# In[56]:


df['LOCATION'][df['CLASS'].isnull()]


# In[57]:


df['CLASS'].value_counts()


# In[58]:


df.isnull().sum()


# In[59]:


df.info()


# In[70]:


df.iloc[104600]


# In[79]:


df.loc[65113,['CLASS']] = 'C+'
df.loc[104600,['CLASS']] = 'C-'
df.loc[106385,['CLASS']] = 'C-'
df.loc[123089,['CLASS']] = 'A'
df.loc[123097,['CLASS']] = 'A'
df.loc[123101,['CLASS']] = 'A'
df.loc[123113,['CLASS']] = 'A'
df.loc[232001,['CLASS']] = 'A'
df.loc[248763,['CLASS']] = 'A'
df.loc[275001,['CLASS']] = 'A'
df.loc[275009,['CLASS']] = 'A'
df.loc[151426,['CLASS']] = 'B'
df.loc[151432,['CLASS']] = 'B'
df.loc[151744,['CLASS']] = 'B'
df.loc[152712,['CLASS']] = 'B'
df.loc[152718,['CLASS']] = 'B'
df.loc[152719,['CLASS']] = 'B'
df.loc[267744,['CLASS']] = 'B'
df.loc[267745,['CLASS']] = 'B'
df.loc[505587,['CLASS']] = 'B'
df.loc[505322,['CLASS']] = 'B'
df.loc[475442,['CLASS']] = 'B'
df.loc[475563,['CLASS']] = 'B'
df.loc[475577,['CLASS']] = 'B'
df.loc[475592,['CLASS']] = 'B'
df.loc[475599,['CLASS']] = 'B'
df.loc[486920,['CLASS']] = 'B'
df.loc[486923,['CLASS']] = 'B'
df.loc[487032,['CLASS']] = 'B'
df.loc[270038,['CLASS']] = 'B'
df.loc[271220,['CLASS']] = 'B'
df.loc[445634,['CLASS']] = 'B'
df.loc[445639,['CLASS']] = 'B'


# In[88]:


df.isnull().sum()


# In[81]:


df['CLASS'].value_counts()


# In[82]:


df.to_csv(r'C:\Users\sofi\Desktop\Data.csv', index = False)


# In[84]:


df[df['ITEM_CODE'].isnull()]


# In[90]:


df.iloc[121665]


# In[86]:


df.loc[121665,['ITEM_CODE']] = '4CHD00H10846BC18H'


# In[89]:


df = df.rename(columns={'BIRTH_YEAR': 'BIRTHDAY'})
df = df.rename(columns={'ITEM_CODE': 'PRODUCT_ID'})
df = df.rename(columns={'CC_LIBELLE_1': 'PRODUCT_NAME'})
df = df.rename(columns={'CC_LIBELLE': 'LABEL'})
df = df.rename(columns={'DESIGNATION': 'COLOR'})
df = df.rename(columns={'PVTTC': 'PRODUCT_PRICE'})
df = df.rename(columns={'PUTTCNET': 'PRODUCT_PRICE_AFTER_REDUCTION'})
df = df.rename(columns={'QTEFACT': 'PRODUCT_QUANTITY'})
df = df.rename(columns={'MLR_REMISE': 'REDUCTION_PERCENT'})
df = df.rename(columns={'GTR_LIBELLE': 'REDUCTION_TYPE'})
df = df.rename(columns={'YX_LIBELLE': 'CUSTOMER_DESCRIPTION'})





# In[93]:


data= pd.read_csv(r'C:\Users\sofi\Downloads\train_gender.csv')
data.info()


# In[97]:


df.isnull().sum()

