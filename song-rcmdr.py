#!/usr/bin/env python
# coding: utf-8

# In[3]:


import turicreate as tc
songs = tc.SFrame('song-data')


# In[4]:


import matplotlib


# In[5]:


songs.head()


# In[6]:


songs.sort('listen_count', ascending= False)[0]


# In[7]:


songs[songs['user_id'] == '50996bbabb6f7857bf0c8019435b5246a0e45cfd']


# In[8]:


songs.show()


# In[9]:


users = songs['user_id'].unique()


# In[10]:


len(users)


# In[11]:


train_data,test_data = songs.random_split(.8, seed =0 )


# In[12]:


popularity_model = tc.popularity_recommender.create(train_data,
                                                   user_id='user_id',
                                                   item_id='song')


# In[13]:


popularity_model.recommend(users = [users[0]])


# In[14]:


personalized_model = tc.item_similarity_recommender.create(train_data, user_id='user_id', item_id='song')


# In[15]:


personalized_model.recommend(users = [users[0]])


# In[16]:


personalized_model.get_similar_items(['Stronger - Kanye West'])


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
model_performance = tc.recommender.util.compare_models(test_data, [popularity_model, personalized_model], user_sample= 0.05)


# In[21]:


##assigment


# In[22]:


west = songs[songs['artist']=='Kanye West']


# In[34]:


foo = songs[songs['artist']=='Foo Fighters']
taylor = songs[songs['artist']=='Taylor Swift']
gaga = songs[songs['artist']=='Lady GaGa']


# In[30]:


len(west['user_id'].unique())


# In[31]:


len(foo['user_id'].unique())


# In[32]:


len(taylor['user_id'].unique())


# In[35]:


len(gaga['user_id'].unique())


# In[41]:


famous_artist = songs.groupby(key_column_names='artist', operations={'total_count': tc.aggregate.SUM('listen_count')})


# In[47]:


famous_artist.sort('total_count', ascending= False)


# In[48]:


famous_artist.sort('total_count', ascending= True)


# In[ ]:




