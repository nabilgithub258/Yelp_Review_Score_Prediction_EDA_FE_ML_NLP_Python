#!/usr/bin/env python
# coding: utf-8

# In[267]:


#####################################################################################################
######################### YELP REVIEWS DATA SET  ####################################################
#####################################################################################################


# In[268]:


#################################################################
############ Part I - Importing
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[269]:


df = pd.read_csv('yelp.csv')


# In[270]:


df.info()


# In[271]:


df.head()


# In[272]:


#####################################################################
########################### Part II - Duplicates
#####################################################################


# In[273]:


df[df.duplicated()]                            #### no duplicates found


# In[274]:


####################################################################
############## Part III - Missing Values
####################################################################


# In[275]:


from matplotlib.colors import LinearSegmentedColormap

Amelia = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])


# In[276]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')                        #### no missing data found

#### why Amelia, if you coming from R then you might have used Amelia package which detects the missing value 
#### On July 2, 1937, Amelia disappeared over the Pacific Ocean while attempting to become the first female pilot to circumnavigate the world


# In[278]:


df.isnull().any()


# In[279]:


####################################################################
############## Part IV - Feature Engineering
####################################################################


# In[280]:


df.head()


# In[281]:


df.type.value_counts()


# In[282]:


df = df[['date','text','stars']]

df.head()                          #### these are the most important cols we will be looking at for this project


# In[283]:


df['dates'] = pd.to_datetime(df.date)


# In[284]:


df.head()


# In[285]:


df.info()


# In[286]:


df = df[['dates','text','stars']]

df.head()


# In[287]:


df.isnull().any()


# In[288]:


df.stars.value_counts()


# In[289]:


df['score'] = df.stars.apply(lambda x: 0 if x in [1,2] else (1 if x == 3 else 2))


# In[290]:


df[['stars','score']].head(10)


# In[291]:


df.score.value_counts()                       #### this will be very helpful, {0:'poor',1:'OK',2:'good'}


# In[292]:


######################################################################
############## Part V - EDA
######################################################################


# In[293]:


df.head()


# In[294]:


x = df.dates[0]

x


# In[295]:


x.day


# In[296]:


x.dayofweek


# In[297]:


x.month


# In[298]:


x.year


# In[299]:


df['day'] = df.dates.apply(lambda x: x.day)


# In[300]:


df['month'] = df.dates.apply(lambda x:x.month)


# In[301]:


df['year'] = df.dates.apply(lambda x:x.year)


# In[302]:


df['day_of_week'] = df.dates.apply(lambda x:x.dayofweek)


# In[303]:


df['month_name'] = df.month.map({1:'Jan',
                         2:'Feb',
                         3:'Mar',
                         4:'Apr',
                         5:'May',
                         6:'Jun',
                         7:'Jul',
                         8:'Aug',
                         9:'Sep',
                         10:'Oct',
                         11:'Nov',
                         12:'Dec'})


# In[304]:


df['day_name'] = df.day_of_week.map({0:'Mon',
                                     1:'Tue',
                                     2:'Wed',
                                     3:'Thr',
                                     4:'Fri',
                                     5:'Sat',
                                     6:'Sun'})


# In[305]:


df.head()                           #### this is amazing, we will get a lot of info from this


# In[306]:


sns.catplot(x='day',data=df,kind='count',hue='score',height=7,aspect=2,palette={0:'red',
                                                                                1:'black',
                                                                                2:'green'})

#### its so close and similar


# In[307]:


sns.catplot(x='day_name',data=df,kind='count',hue='score',height=7,aspect=2,palette={0:'red',
                                                                                1:'black',
                                                                                2:'green'})

#### its very close and tight but it seems like we get better ratings on Mondays
#### worst ratings are from Sundays but again its very tight


# In[308]:


pl = sns.FacetGrid(df,hue='score',aspect=4,height=4,palette={ 0:'red',
                                                                    1:'black',
                                                                    2:'green'})

pl.map(sns.kdeplot,'month',fill=True)

pl.set(xlim=(0,df.month.max()))

pl.add_legend()

#### again nothing to see here, its very close but we see some good ratings in the month of Jan-Mar


# In[309]:


pl = sns.FacetGrid(df,hue='score',aspect=4,height=4,palette={ 0:'red',
                                                                    1:'black',
                                                                    2:'green'})

pl.map(sns.kdeplot,'year',fill=True)

pl.set(xlim=(df.year.min(),df.year.max()))

pl.add_legend()

#### seems like the majority of our data is from 2010-2012 which makes sense due to the advancement of internet


# In[310]:


df.month_name.value_counts()


# In[311]:


heat = df.groupby(['year','month_name','day_name'])['score'].sum().unstack().unstack().fillna(0)

heat


# In[312]:


fig, ax = plt.subplots(figsize=(25,12))

sns.heatmap(heat,ax=ax,linewidths=0.5,annot=True,cmap='viridis')


# In[313]:


df[df.year == 2013]                 #### the data from this year is more concerning, one reason may be because we dont have much density to this year which skews the outcome


# In[314]:


df[df.year == 2013]['score'].plot(legend=True,figsize=(20,7),marker='o',markersize=14,markerfacecolor='black',linestyle='dashed',linewidth=4,color='red')

#### seems like majority is good scores


# In[315]:


df.groupby('month_name')['score'].sum().plot(legend=True,figsize=(20,7),marker='o',markersize=14,markerfacecolor='black',linestyle='dashed',linewidth=4,color='red')

#### the best month is Jan for reviews


# In[316]:


df.groupby('month_name')['score'].sum()               #### worst is November


# In[317]:


heat = df[df['year'].isin([2008,2009,2010,2011,2012])].groupby(['year','month_name','day'])['score'].sum().unstack().unstack().fillna(0)

heat              #### we only taking years 2008-2012 where the majority of reviews came through


# In[318]:


fig, ax = plt.subplots(figsize=(40,22))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis')

#### the one that stands out is 2011


# In[319]:


heat = df[df['year'].isin([2008,2009,2010,2011,2012])].groupby(['year','month_name'])['score'].sum().unstack().fillna(0)

heat


# In[320]:


fig, ax = plt.subplots(figsize=(25,12))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis',annot=True)

#### here we see that Jan of 2012 which has the most densed reviews


# In[321]:


heat[heat == heat.loc[2012].max()].loc[2012].loc['Jan']           #### this the most densed one


# In[322]:


df.head()


# In[323]:


df['length'] = df.text.apply(len)


# In[324]:


df.head()                        #### this gives us more holistic idea


# In[325]:


df['length'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Yelp Length Graph')

plt.xlabel('Number of people')

plt.ylabel('Density')

#### interesting, we do see some outliers


# In[326]:


df.length.mean()


# In[327]:


df.length.max()


# In[328]:


df.length.quantile(0.99)            #### this is what we seeing, amazing


# In[329]:


custom = {0:'red',
          1:'black',
          2:'green'}

plt.figure(figsize=(17,5))
sns.histplot(x='length',data=df,hue='score',palette=custom,multiple='dodge',bins=5)

#### seems like people who write lengthy reviews are the ones who leave a better reviews compared to others, I wasn't expecting this honestly


# In[330]:


df = df[df.length <= 3000]                        #### we dont want outliers for this


# In[331]:


df.head()


# In[332]:


df.info()


# In[333]:


heat = df.groupby(['length'])['score'].sum().sort_values(ascending=False).head(20)

heat


# In[334]:


heat.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=15,linestyle='dashed',linewidth=4)

plt.title('Yelp Heat Graph')

plt.xlabel('Review Length')

plt.ylabel('Density')

plt.savefig('Yelp_heat_custom_lineplot.jpeg', dpi=300, bbox_inches='tight')

#### it depicts the length of reviews according to the score values, here we see words 200-250 has the most count


# In[335]:


custom = {0:'red',
          1:'black',
          2:'green'}

plt.figure(figsize=(17,5))
sns.histplot(x='length',data=df,hue='score',palette=custom,multiple='dodge',bins=5)

#### we doing the same on the new df and now we see a much better picture 


# In[336]:


g = sns.jointplot(x='length',y='score',data=df,kind='kde',fill=True,color='red')

g.fig.set_size_inches(17,9)

#### again we see good reviews are heavily densed at around 200 words


# In[337]:


g = sns.jointplot(x='length',y='month',data=df,kind='reg',x_bins=[range(1,3000)],color='black',joint_kws={'line_kws':{'color':'red'}})

g.fig.set_size_inches(17,9)

g.ax_joint.set_xlim(0,3000)
g.ax_joint.set_ylim(0,df.month.max())

#### no correlation


# In[338]:


g = sns.jointplot(x='length',y='score',data=df,kind='reg',x_bins=[(1,5,10,20,40,50,70,100,140,200,300,400,500,600,700,900,1200,1500,2000,2200,2500,2700,2900,3000)],color='black',joint_kws={'line_kws':{'color':'red'}})

g.fig.set_size_inches(17,9)

g.ax_joint.set_xlim(0,3000)
g.ax_joint.set_ylim(0,df.score.max())

#### more words are somehow related to poor reviews


# In[339]:


custom = {0:'red',
          1:'black',
          2:'green'}

pl = sns.FacetGrid(df,hue='score',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'day',fill=True)

pl.set(xlim=(0,df.day.max()))

pl.add_legend()

#### seems like 15-20 days we get bad reviews


# In[340]:


pl = sns.FacetGrid(df,hue='month_name',aspect=4,height=4,palette='Dark2')

pl.map(sns.kdeplot,'length',fill=True)

pl.set(xlim=(0,3000))

pl.add_legend()

#### we dont see much difference honestly


# In[341]:


pl = sns.FacetGrid(df,hue='month_name',aspect=4,height=4,palette='Dark2')

pl.map(sns.kdeplot,'score',fill=True)

pl.set(xlim=(0,df.score.max()))

pl.add_legend()

#### Jan being the most reviewed and most of them being positive


# In[342]:


g = sns.lmplot(x='length',y='day_of_week',data=df,x_bins=[(0,10,20,50,100,150,170,200,250,300,400,500,700,900,1200,1500,1800,2000,2200,2500,2700,2900,3000)],height=7,aspect=2,line_kws={'color':'red'},scatter_kws={'color':'black'})

#### not much we see here obviously


# In[343]:


df['dates'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Yelp Dates Graph')

plt.xlabel('Number of people')

plt.ylabel('Year')

#### we see the majority of reviews are from 2009-2013, we will now be moving to the model phase


# In[168]:


######################################################################
############## Part VI - Model - Classification
######################################################################


# In[199]:


df.head()


# In[200]:


X = df.drop(columns=['dates','text','month_name','day_name','stars','score'])

X.head()


# In[201]:


y = df.score

y.head()


# In[202]:


y.value_counts()                               #### this will be a problem due to low density of data and to top it off its very imbalanced


# In[203]:


#### now we will do by taking care of any multicollinearity

from statsmodels.tools.tools import add_constant

X_with_constant = add_constant(X)

X_with_constant.head()                    #### setting up Vif


# In[204]:


vif = pd.DataFrame()


# In[205]:


vif["Feature"] = X_with_constant.columns


# In[206]:


X_with_constant.isnull().any()


# In[207]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif["VIF"] = [variance_inflation_factor(X_with_constant.values, i) for i in range(X_with_constant.shape[1])]


# In[208]:


vif                          #### seems good


# In[209]:


X.head()


# In[210]:


X = df.text

X.head()


# In[211]:


y.head()


# In[213]:


X.isnull().any()


# In[214]:


y.isnull().any()


# In[232]:


from sklearn.model_selection import train_test_split


# In[233]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[234]:


from sklearn.pipeline import Pipeline


# In[235]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[236]:


model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])


# In[237]:


model.fit(X_train, y_train)


# In[238]:


y_predict = model.predict(X_test)


# In[239]:


from sklearn import metrics


# In[240]:


print(metrics.classification_report(y_test,y_predict))                #### not good


# In[241]:


from sklearn.ensemble import RandomForestClassifier


# In[242]:


model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', RandomForestClassifier(random_state=42,n_jobs=-1,class_weight='balanced'))
])


# In[243]:


model.fit(X_train, y_train)


# In[244]:


y_predict = model.predict(X_test)


# In[245]:


print(metrics.classification_report(y_test,y_predict))                       #### still not good on imbalanced targets


# In[246]:


from imblearn.over_sampling import SMOTE


# In[247]:


from imblearn.pipeline import Pipeline as ImbPipeline


# In[248]:


model = ImbPipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1))
])


# In[249]:


model.fit(X_train, y_train)


# In[250]:


y_predict = model.predict(X_test)


# In[251]:


print(metrics.classification_report(y_test,y_predict))


# In[252]:


model = ImbPipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1,class_weight='balanced'))
])


# In[253]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[254]:


y_predict = model.predict(X_test)


# In[255]:


print(metrics.classification_report(y_test,y_predict))                 #### some improvements


# In[256]:


from lightgbm import LGBMClassifier


# In[258]:


model = ImbPipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('smote', SMOTE(random_state=42)),
    ('clf', LGBMClassifier(class_weight='balanced', random_state=42,n_jobs=-1))

])


# In[259]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[260]:


y_predict = model.predict(X_test)


# In[261]:


print(metrics.classification_report(y_test,y_predict))                      #### much better


# In[262]:


from catboost import CatBoostClassifier


# In[263]:


model = ImbPipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('smote', SMOTE(random_state=42)),
    ('clf', CatBoostClassifier(auto_class_weights='Balanced', random_state=42))
])


# In[264]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[265]:


y_predict = model.predict(X_test)


# In[266]:


print(metrics.classification_report(y_test,y_predict))                   #### not much improvements


# In[ ]:


############################################################################################################################
#### We will be concluding our modeling phase at this point due to the limited improvement in model performance. ###########
#### Despite applying various techniques to address class imbalance using ImbPipeline with SMOTE, the results have #########
#### not significantly improved. Our efforts included extensive preprocessing, feature engineering, and the use of #########
#### advanced models such as LightGBM and CatBoost, which yielded the best results. However, further enhancements in #######
#### metrics and performance have plateaued. ###############################################################################
############################################################################################################################

