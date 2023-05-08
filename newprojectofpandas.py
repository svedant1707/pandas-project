#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv(r'C:\Users\sveda\OneDrive\Desktop\Journals.csv')
df.head(10)


# In[3]:


df.tail(15)


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull()


# In[7]:


df.isnull().sum()


# In[8]:


df.loc[::6,'title':'publisher']


# In[9]:


df.iloc[1:10]


# In[10]:


#dtypes
df.pages.dtypes


# In[11]:


df.dtypes


# In[12]:


#cumsum() It provides you with the cumulative sum

df[['foundingyear','publisher']].cumsum()


# In[13]:


#df.sample
df.sample(n=10)


# In[14]:


#df.where()

df['price'].where(df['price']>250)


# In[15]:


#df.unique()

df.field.unique()


# In[16]:


#df.rank()
df['field'].rank()


# In[17]:


#fillna
df['charpp'].fillna(df['charpp'].mean(), inplace=False)


# In[18]:


#group by 
df.groupby('publisher')['price'].sum()


# In[19]:


#pct_change()
df.citations.pct_change()


# In[20]:


#df.count()
df.count(1)


# In[21]:


#value_count()
df['foundingyear'].value_counts()


# In[22]:


#crosstab()

pd.crosstab(df['publisher'],df['foundingyear'])


# In[23]:


#isin()
publisher=['Blackwell','Cambridge Univ Pres','Carfax','Ec. Society of Australia']
df[df.publisher.isin(publisher)]


# In[24]:


#to_transpose()

df_transposed = df.transpose()
df_transposed.head()


# In[25]:


#df.assign()
df_new = df.assign(title=df['price']+5)
df_new.head()


# In[26]:


#t.head
df.describe().T.head(10)


# In[27]:


df.describe().T.drop('count',axis=1).style.highlight_max(color='darkred')


# In[28]:


df.describe().T.drop('count',axis=1).style.background_gradient(subset=['mean','50%'],cmap='Reds')


# In[29]:


#squeeze
df.squeeze('columns')


# In[30]:


#pd.pivot_table()

pivot_table=pd.pivot_table(df,index='publisher',values='price',aggfunc='sum')
pivot_table.head()


# In[31]:


df['publisher'].value_counts().plot(kind='pie')


# In[32]:


plt.figure(figsize=(20,5))
sns.relplot(x='title',y='price',data=df,kind='line')


# In[33]:


#scatter plot

plt.scatter(df['citations'],df['pages'])
plt.title('Scatter plot')
plt.xlabel('citations')
plt.ylabel('pages')
plt.show()


# In[34]:


sns.relplot(x='price',y='pages',data=df,hue='title')


# In[35]:


df['foundingyear'].plot(kind='hist')
plt.show()


# In[36]:


sns.boxplot(x='subs', data=df)
plt.show()


# In[37]:


#Heatmap: A heatmap is a plot that shows the correlation between different variables.

sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
plt.show()


# In[38]:


#bar plot A bar plot is a plot that shows the comparison between different categories.

df['publisher'].value_counts().plot(kind='bar')
plt.show()


# In[39]:


#box plot A box plot is a plot that shows the distribution of a variable and its outliers.

df.boxplot(column='subs')
plt.show()


# In[40]:


#area plot An area plot is a plot that shows the cumulative totals of a variable over time
df.plot(kind='area', x='publisher', y='price')
plt.show()


# In[41]:


#Hexbin plot: A hexbin plot is a plot that shows the density of a two-dimensional dataset.
df.plot(kind='hexbin', x='subs', y='price', gridsize=25)
plt.show()


# In[42]:


#Stacked bar plot: A stacked bar plot is a plot that shows the comparison between different categories with multiple variables

df.groupby('charpp')['title', 'price'].sum().plot(kind='bar', stacked=True)
plt.show()


# In[43]:


#Kernel density estimation (KDE) plot: A KDE plot is a plot that shows the density of a variable.

sns.kdeplot(df['price'])
plt.show()


# In[44]:


#Pairplot: A pairplot is a plot that shows the relationship between multiple variables.

sns.pairplot(df)
plt.show()


# In[45]:


#Violin plot: A violin plot is a plot that shows the distribution of a variable.

sns.violinplot(y='subs', data=df)
plt.show()


# In[46]:


#Swarm plot: A swarm plot is a plot that shows the relationship between two variables.

sns.swarmplot(x='publisher', y='foundingyear', data=df)
plt.show()


# In[47]:


#line  plot with multiplw lines
df.plot(x='title', y=['foundingyear', 'price'])
plt.show()


# In[48]:


#sub plots

fig, axs = plt.subplots(nrows=2, ncols=2)
df['charpp'].plot(ax=axs[0,0])
df['pages'].plot(ax=axs[0,1])
df['price'].plot(ax=axs[1,0])
df['subs'].plot(ax=axs[1,1])
plt.show()


# In[69]:


sns.distplot(df)


# Q1.show me the data  of kluwer

# In[63]:


kluwer_data = df[df['publisher'] == 'Kluwer']
print(kluwer_data)


# Q2.

# In[64]:


page = df[df['pages'] >100]
print(page['Blackwell'])


# Q3.What is the maximum and minimum price of all books in the 'Journals.csv' file?

# In[53]:


max_price = df['price'].max()
min_price = df['price'].min()

print(f"The maximum price of all books is: {max_price}")
print(f"The minimum price of all books is: {min_price}")


# Q4.How many books were published by the publisher 'kluwer'?

# In[57]:


Blackwell_books = df[df['publisher'] == 'Blackwell']

num_Blackwell_books = Blackwell_books.shape[0]

print(f"The number of books published by 'Blackwell' is: {num_Blackwell_books}")


# Q5.What is the average page number of all books in the 'Journals.csv' file?
# 

# In[62]:


avg_page_num = df['pages'].mean()

print(f"The average page number of all books is: {avg_page_num}")


# Q6.highest sells of book

# In[ ]:


max_sales_book = df.loc[df['Sales'].idxmax()]

print(f"The book with the highest sales is:\n{max_sales_book}")


# Q7.how to get the information of foundingyear above 1972
# 
# 

# In[68]:


mask = df['foundingyear'] > 1972
filtered_df = df[mask]
print(filtered_df.info())


# In[ ]:




