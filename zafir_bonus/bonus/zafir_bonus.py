#!/usr/bin/env python
# coding: utf-8

# In[374]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.util import bigrams,trigrams
from fuzzywuzzy import fuzz
from collections import defaultdict
import re
from textblob import TextBlob
import json

nltk.download('punkt')
nltk.download('stopwords')


# In[375]:

year = input("Please input year")
pd.set_option('display.max_colwidth', None)
tweet_dataset=f'../../data/gg{year}.json'
df=pd.read_json(tweet_dataset)


# In[376]:


df


# In[377]:


def filter_rows_by_keywords(df, column_name, keywords):
    
    pattern = '|'.join(keywords)
    return df[df[column_name].str.contains(pattern, case=False, na=False)]


# In[378]:


keywords=['best']
filtered_df = filter_rows_by_keywords(df, 'text', keywords)
filtered_df.head()


# In[379]:


#bigram or trigram, variable written as bigram only

text = " ".join(row for row in filtered_df['text']).lower()


tokens = word_tokenize(text)


filtered_tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]


bigram_list = list(trigrams(filtered_tokens))

best_bigrams = [bg for bg in bigram_list if bg[0] == "best"]


best_bigram_counts = Counter(best_bigrams)


top_best_bigrams = best_bigram_counts.most_common(26)

print(top_best_bigrams)


# In[380]:


df_movie_actor=pd.read_json(f"../../eric/data/gg{year}_sorted_annotated.json")
df_movie_actor


# In[381]:


bigram_patterns = [bigram for bigram, count in top_best_bigrams]

dataframe_actor_or_filtered=df_movie_actor

def contains_top_bigram(text):
    tokens = word_tokenize(text.lower())
    text_bigrams = list(trigrams(tokens))
    return any(bg in text_bigrams for bg in bigram_patterns)

filtered_df_best_words = dataframe_actor_or_filtered[dataframe_actor_or_filtered['text'].apply(contains_top_bigram)]


# In[382]:


filtered_df_best_words


# In[383]:


filtered_df_best_words


# In[ ]:





# In[384]:


filtered_df_best_words[['text']]


# In[385]:


#tried grouping similar tweets

# def assign_groups_to_tweets(df, trigrams, threshold=70):
#     groups = defaultdict(list)
#     group_counter = 0
    
#     for index, row in df.iterrows():
#         tweet = row['text'].lower()
#         tweet = re.sub(r'[^a-zA-Z0-9\s]', ' ', tweet)
#         for trigram in trigrams:
#             tri_text = ' '.join(trigram[0])
#             if tri_text in tweet:
#                 #print(tri_text,tweet)
#                 candidate = ' '.join(tweet.split()[tweet.split().index(trigram[0][0]):])
#                 added = False
#                 for key in groups.keys():
#                     if fuzz.partial_ratio(candidate, key) > threshold:
#                         groups[key].append(candidate)
#                         df.at[index, 'group'] = key
#                         added = True
#                         break
#                 if not added:
#                     group_counter += 1
#                     group_key = f'Group_{group_counter}'
#                     groups[group_key].append(candidate)
#                     df.at[index, 'group'] = group_key

#     return df, groups



# In[386]:


# filtered_tweets = filtered_df_best_words.copy()
# filtered_tweets['text'] = filtered_tweets['text'].apply(lambda x: x.lower())
# filtered_tweets['group'] = None 

# trigrams = top_best_bigrams

# filtered_tweets, groups = assign_groups_to_tweets(filtered_tweets, trigrams)


# # print(filtered_tweets.head())
# for key, group in groups.items():
#     print(f"{key}:")
#     for candidate in set(group):
#         print(f"  - {candidate}")


# In[387]:


def assign_trigrams_to_tweets(df, trigrams):
    df['matching_trigram'] = None  
    
    for index, row in df.iterrows():
        tweet = row['text'].lower()
        for trigram in trigrams:
            tri_text = ' '.join(trigram[0])
            if tri_text in tweet:
                df.at[index, 'matching_trigram'] = tri_text
                break 

    return df


# In[388]:


grouped_awards = assign_trigrams_to_tweets(filtered_df_best_words,top_best_bigrams)


# In[389]:


grouped_awards


# In[390]:


# grouped_awards[grouped_awards['matching_trigram']=='best television series'].head(50)


# In[391]:


# grouped_awards.groupby('matching_trigram').count()


# In[392]:


def remove_strings(list_to_modify, reference_list, special_string):
    special_string = special_string.lower()
    
    list_to_modify = [string for string in list_to_modify 
                      if string.lower() not in (ref_string.lower() for ref_string in reference_list)]
    
    list_to_modify = [string for string in list_to_modify 
                      if special_string.find(string.lower()) == -1]
    
    return list_to_modify


# In[393]:


def remove_actor_names(list_to_modify, actors_list):
    names_to_remove = []
    for sublist in actors_list:
        for _, name in sublist:
            if len(name.split())==2:
                first_name, last_name = name.split()
                names_to_remove.extend([first_name.lower(), last_name.lower(), name.lower().replace(" ", "")])
            elif len(name.split())==1:
                first_name = name.split()[0]
                names_to_remove.extend([first_name.lower()])
            
    list_to_modify = [string for string in list_to_modify
                      if string.lower() not in names_to_remove]
    
    return list_to_modify


# In[394]:


def find_winner(df,movie_or_actor,award_name):
    actor_names = []


    for index, row in df.iterrows():
        actors_list = row[movie_or_actor]
        for actor in actors_list:
            name = actor[1]
            actor_names.append(name)
    #print(actor_names)
    if movie_or_actor=='movie':
        ref_list=['Drama','actor']
        actor_names=remove_strings(actor_names,ref_list,award_name)
        actor_names=remove_actor_names(actor_names,df['name'])
    #print(actor_names)  
    actor_counts = Counter(actor_names)

    max_occurrence = max(actor_counts.values())

    most_frequent_actors = [actor for actor, count in actor_counts.items() if count == max_occurrence]
    
    filtered_list=[]
    for string in most_frequent_actors:
        if not any(string in s for s in most_frequent_actors if string != s):
            filtered_list.append(string)
    return filtered_list


# In[395]:


# winner = find_winner(grouped_awards[grouped_awards['matching_trigram']=='best supporting actress'],movie_or_actor='name',award_name='best supporting actress')
# print(winner)


# In[396]:


def get_presenters(df, keywords=['presenting', 'presenter', 'announce', 'announcing']):
    
    filtered_df = df[df['text'].str.contains('|'.join(keywords), case=False, na=False)]
    
    
    actor_names = set()
    for index, row in filtered_df.iterrows():
        actors_list = row['name']
        for actor in actors_list:
            name = actor[1]
            actor_names.add(name)
            
    return list(actor_names)


# In[397]:



# presenters = get_presenters(grouped_awards[grouped_awards['matching_trigram']=='best actor tv'])
# print(presenters)


# In[398]:


def get_nominees(df,movie_or_actor, keywords=['nominee','nominees','nominated'],winner=[''],award_name=''):
    
    filtered_df = df[df['text'].str.contains('|'.join(keywords), case=False, na=False)]
    
    #filtered_df = df[df['text'].str.contains('|'.join(winner), case=False, na=False)]
    
    actor_names = []
    for index, row in filtered_df.iterrows():
        actors_list = row[movie_or_actor]
        for actor in actors_list:
            name = actor[1]
            actor_names.append(name)
    #print(actor_names)
    if movie_or_actor=='movie':
        ref_list=['Drama','actor']
        actor_names=remove_strings(actor_names,ref_list,award_name)
        actor_names=remove_actor_names(actor_names,df['name'])
        
    actor_counts = Counter(actor_names)
    #print(actor_counts)

    #max_occurrence = max(actor_counts.values())
    
    most_frequent_actors = sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)
    most_frequent_actors=list(map(lambda x: x[0], most_frequent_actors[:4]))
    
    filtered_list=[]
    for string in most_frequent_actors:
        if not any(string in s for s in most_frequent_actors if string != s):
            filtered_list.append(string)
    return filtered_list


# In[399]:


# print(get_nominees(df_movie_actor,movie_or_actor='name',winner=winner,award_name='best supporting actress'))


# In[ ]:





# In[400]:


def audit_nominees(df, keywords=['presenting', 'presenter', 'announce', 'announcing']):
    
    filtered_df = df[df['text'].str.contains('|'.join(keywords), case=False, na=False)]
    return filtered_df


# In[401]:


# temp_df=audit_nominees(df_movie_actor,["democrat"])
# temp_df.head(50)


# In[402]:


# temp_presenters=get_presenters(temp_df,keywords=['nominee', 'nominees'])
# temp_presenters


# In[403]:


award_list=grouped_awards['matching_trigram'].unique()
award_list


# In[404]:


for i in award_list:
    print(i)


# In[405]:


awards={}
filtered_awards={}
potential_movie_awards=['series','film']

for award_name in award_list:
    movie_or_actor="name"
    if any(word in potential_movie_awards for word in award_name.split()):
        movie_or_actor="movie"
        
    winner = find_winner(grouped_awards[grouped_awards['matching_trigram']==award_name],movie_or_actor=movie_or_actor,award_name=award_name)
    winner_dict={}
    winner_dict["winner"]=winner
    awards[award_name]=winner_dict
    
    nominee=get_nominees(df_movie_actor,movie_or_actor=movie_or_actor,winner=winner,award_name=award_name)
    nominee_dict={}
    nominee_dict["nominees"]=nominee
    awards[award_name].update(nominee_dict)
    
    presenters=get_presenters(grouped_awards[grouped_awards['matching_trigram']==award_name])
    presenter_dict={}
    presenter_dict["Presenter"]=presenters
    awards[award_name].update(presenter_dict)
    


# In[406]:


awards


# In[407]:


from textblob import TextBlob


grouped_awards['names'] = grouped_awards['name'].apply(lambda x: [i[1].lower() for i in x])

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity


grouped_awards['sentiment'] = grouped_awards['text'].apply(get_sentiment)

df_movie_actor['names'] = df_movie_actor['name'].apply(lambda x: [i[1].lower() for i in x])



df_movie_actor['sentiment'] = df_movie_actor['text'].apply(get_sentiment)


actor_sentiment = {}

for index, row in df_movie_actor.iterrows():
    for name in row['names']:
        
        first_name, last_name = name.split()[0], name.split()[-1]
        
        
        if first_name.lower() in row['text'].lower() or last_name.lower() in row['text'].lower():
            if name not in actor_sentiment:
                actor_sentiment[name] = 0
            actor_sentiment[name] += row['sentiment']


sorted_actors = sorted(actor_sentiment.items(), key=lambda x: x[1], reverse=True)

loved_actor_names = list(map(lambda x: x[0], sorted_actors[:3]))
hated_actor_names = list(map(lambda x: x[0], sorted_actors[-3:]))

print("Top 3 Loved Actors:", loved_actor_names)
print("Top 3 Hated Actors:", hated_actor_names)


# In[408]:


awards['Top 3 Loved Actors']=loved_actor_names
awards['Top 3 Hated Actors']=hated_actor_names

filtered_awards['Top 3 Loved Actors']=loved_actor_names
filtered_awards['Top 3 Hated Actors']=hated_actor_names


# In[409]:


awards


# In[410]:


# grouped_awards


# In[423]:


joke_keywords = ['lol', 'haha', 'funny', 'joke', 'lmao', 'rofl']

def contains_keyword(text, keywords):
    return any(keyword in text.lower() for keyword in keywords)


jokes_df = df_movie_actor[
    (df_movie_actor['text'].apply(lambda x: contains_keyword(x, joke_keywords))) & 
    (df_movie_actor['sentiment'] > 0.5)
]


jokes_df[['text', 'user', 'names', 'sentiment']]







# In[412]:


joke_keywords_expressions = ['lol', 'haha']
expressions= ['face','expression','humiliation']
humiliation_df = df_movie_actor[
    (df_movie_actor['text'].apply(lambda x: contains_keyword(x, joke_keywords))) & 
    (df_movie_actor['sentiment'] > 0.5)
]
humiliation_df = humiliation_df[
    (humiliation_df['text'].apply(lambda x: contains_keyword(x, expressions))) & 
    (humiliation_df['sentiment'] > 0.5)
]
def split_text(text, keywords):
    for keyword in keywords:
        before_keyword, keyword, after_keyword = text.partition(keyword)
        if keyword:
            return pd.Series([before_keyword.strip(), after_keyword.strip()])

    return pd.Series([None, None])


humiliation_df[['prior_col', 'post_col']] = humiliation_df['text'].apply(lambda x: split_text(x, expressions))


def clean_text(text):
    try:
        return text.lower().replace("'s", '')
    except:
        return ''


def check_names(row):
    names = [clean_text(name) for name in row['names']]
    prior_text = clean_text(row['prior_col'])
    post_text = clean_text(row['post_col'])

    for name in names:
        if name in prior_text:
            row['prior_person'] = name
        if name in post_text:
            row['post_person'] = name
            
    return row

# Applying the function
humiliation_df = humiliation_df.apply(check_names, axis=1)
prior_person=humiliation_df['prior_person'].mode()[0]
post_person=humiliation_df['post_person'].mode()[0]
award_name="most_humiliated_actor_due_to_another_actor"
most_humiliated= prior_person + ' was the most humiliated person in the award show because of ' + post_person
awards[award_name]=most_humiliated
filtered_awards[award_name]=most_humiliated


# In[413]:


#awards


# In[414]:


#find people with best jokes
def match_pattern(text):
    pattern = r'best.*joke'
    return bool(re.search(pattern, text, re.IGNORECASE))

potential_df=grouped_awards[grouped_awards['sentiment']> 0.5]
mask = potential_df['text'].apply(match_pattern)
matching_df = potential_df[mask]
concatenated_list = []

for index, row in matching_df.iterrows():
    concatenated_list.extend(row['names'])

unique_set = set(concatenated_list)

unique_list = list(unique_set)
awards['funniest jokes by']=unique_list
filtered_awards['funniest jokes by']=unique_list


# In[415]:


awards


# In[416]:


parties = ['Democrat', 'Republican', 'afterparty']


party_analysis = pd.DataFrame(columns=['party', 'average_sentiment', 'mention_count'])

for party in parties:
    
    party_mentions = df_movie_actor[df_movie_actor['text'].str.contains('|'.join([party]), case=False, regex=True)]
    
    
    average_sentiment = party_mentions['sentiment'].mean()
    mention_count = len(party_mentions)
    
    
    party_analysis = party_analysis.append({'party': party,
                                            'average_sentiment': average_sentiment,
                                            'mention_count': mention_count}, ignore_index=True)


party_analysis = party_analysis.sort_values(by='mention_count', ascending=False)
print(party_analysis)
party_names=' '
for party_name in party_analysis['party']:
    party_names= party_names +party_name +", "
party_analysis['mention_count'] = party_analysis['mention_count'].astype(float)
party_analysis['average_sentiment'] = party_analysis['average_sentiment'].astype(float)    
party_analysis_string_name='Following parties were discussed' + party_names

highest_party_count = party_analysis.loc[party_analysis['mention_count'].idxmax()]['party']
lowest_party_count = party_analysis.loc[party_analysis['mention_count'].idxmin()]['party']


highest_party_sentiment = party_analysis.loc[party_analysis['average_sentiment'].idxmax()]['party']
lowest_party_sentiment = party_analysis.loc[party_analysis['average_sentiment'].idxmin()]['party']


# print(f"The party with the highest mention count is: {highest_party_count}")
# print(f"The party with the lowest mention count is: {lowest_party_count}")
# print(f"The party with the highest average sentiment is: {highest_party_sentiment}")
# print(f"The party with the lowest average sentiment is: {lowest_party_sentiment}")
#print(party_analysis_string_name)
parties_analysis_dict={}
parties_analysis_dict["Parties discussed"]=party_analysis_string_name
parties_analysis_dict["The party with the highest mention count is"]=highest_party_count
parties_analysis_dict["The party with the lowest mention count is"]=lowest_party_count
parties_analysis_dict["The party with the highest average sentiment is"]=highest_party_sentiment
parties_analysis_dict["The party with the lowest average sentiment is"]=lowest_party_sentiment

awards['Parties analysis']=parties_analysis_dict
filtered_awards['Parties analysis']=parties_analysis_dict


# In[417]:


awards


# In[418]:


# json_file='temp_json_awards_delete.json'
# with open(json_file, 'w', encoding='utf-8') as file:
#     json.dump(awards, file, ensure_ascii=False, indent=4)


# In[419]:


performer_keywords=['controversial']
performers_df = df_movie_actor[df_movie_actor['text'].str.contains('|'.join(performer_keywords), case=False, na=False)]
performers_df


# In[420]:


performers_df = df_movie_actor[df_movie_actor['text'].str.contains(r'controversial.*speech', regex=True, case=False, na=False)]
name_counts = Counter(name for names_list in performers_df['names'].dropna() for name in names_list if name != 'speech')
if name_counts:
    # Finding the most common name
    most_common_name, most_common_count = name_counts.most_common(1)[0]
    awards['most controversial speech by']=most_common_name
    filtered_awards['most controversial speech by']=most_common_name


# In[ ]:


# In[421]:


json_file='all_json_awards_temp.json'
with open(json_file, 'w', encoding='utf-8') as file:
    json.dump(awards, file, ensure_ascii=False, indent=4)


# In[422]:


json_file='final_json_awards_bonus.json'
with open(json_file, 'w', encoding='utf-8') as file:
    json.dump(filtered_awards, file, ensure_ascii=False, indent=4)

