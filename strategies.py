import nltk
from pathlib import Path
from typing import Dict, Any, Optional, List
import ahocorasick
from multiprocessing import Pool
import pickle
from tqdm import tqdm
import nltk
import pandas as pd
from utils import Award_Category
import regex as re
from utils import extract_movie, extract_name
import spacy
nlp = spacy.load("en_core_web_sm")

class Strategy:
    def __init__(self, name: str, requirements: Optional[List[str]] = None):
        self.name = name
        self.winner_regex = r"(won|wins|winner|goes to|acceptance speech|congrats|congratulations|congratulate)"
        self.nominee_regex = r"(nominations|nominee|nominated|rob|robbing|robbed|should\'ve been|should\'ve won)"
        self.presenter_regex = r"(present|presented|presents|presenting)"
        self.leftwords = ["wins","won","nominated","present"]
        self.rightwords = ["congrats","congratulations","robbing","goes to"]
        self.bothwords = ["winner","winner","acceptance speech", "congratulate","should've", "robbed","rob","nominations","nominee","presented","presents"]

    def get_keyword_splits(self, text: str):
        winner_split = re.findall(r'(.+)\b' + self.winner_regex + r'\b(.+)', text)
        nominee_split = re.findall(r'(.+)\b' + self.nominee_regex + r'\b(.+)', text)
        presenter_split = re.findall(r'(.+)\b' + self.presenter_regex + r'\b(.+)', text)
        return winner_split, nominee_split, presenter_split
        
    def extract(self, data: List[Dict[str, Any]]) -> List[str]:
        raise NotImplementedError
    
    def single_strategy_cluster(self, data: List[Dict[str, Any]]) -> List[str]:
        raise NotImplementedError

class FirstPass(Strategy):
    pass

# Full regex matching and syntactic tree to find awards and all other entities
class SemanticParseWithoutAward(Strategy):
    def __init__(self, name:str, timebox_award: Award_Category):
        super().__init__("KeywordRegex" if name is None else name)
        self.timebox_award = timebox_award
        self.presenters = {}
        self.nominees = {}
        self.winners = {}
        
    def extract(self, tweet: Dict):
        text = tweet['clean_text'].lower()
        winner_split, nominee_split, presenter_split = self.get_keyword_splits(text)
        splits = [presenter_split,nominee_split, winner_split]
        nominee_found = False
        for i in range(len(splits)):
            if nominee_found:
                break
            if i == 0:
                names = extract_name(tweet)
                if len(names) == 1:
                    self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[0] in self.presenters else 1
                else:
                    index = text.find(splits[i][1])
                    split_word = splits[i][1]
                    for name in names:
                        if split_word in self.leftwords and name in text[:index]:
                            self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[0] in self.presenters else 1
                        elif split_word in self.rightwords and name in text[index:]:
                            self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[0] in self.presenters else 1
                        elif split_word in self.bothwords:
                            self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[0] in self.presenters else 1
                continue
                    
            if self.timebox_award.entity_type == "person":
                entities = extract_name(tweet)
                
            elif self.timebox_award.entity_type == "movie":
                entities = extract_movie(tweet, self.award)
                if len(entities) == 1:
                    if i == 1:
                        self.nominees[entities[0]] = self.nominees[entities[0]] + 1 if entities[0] in self.nominees else 1
                        nominee_found = True
                    elif i == 2:
                        self.winners[entities[0]] = self.winners[entities[0]] + 1 if entities[0] in self.winners else 1
                else:
                    index = text.find(splits[i][1])
                    split_word = splits[i][1]
                    for entity in entities:
                        if split_word in self.leftwords and entity in text[:index]:
                            if i == 0:
                                self.nominees[entity] = self.nominees[entity] + 1 if entity in self.nominees else 1
                                nominee_found = True

                            elif i == 1:
                                self.winners[entity] = self.winners[entity] + 1 if entity in self.winners else 1
                        elif split_word in self.rightwords and entity in text[index:]:
                            if i == 0:
                                self.nominees[entity] = self.nominees[entity] + 1 if entity in self.nominees else 1
                                nominee_found = True
                            elif i == 1:
                                self.winners[entity] = self.winners[entity] + 1 if entity in self.winners else 1
                        elif split_word in self.bothwords:
                            if i == 0:
                                self.nominees[entity] = self.nominees[entity] + 1 if entity in self.nominees else 1
                                nominee_found = True
                            elif i == 1:
                                self.winners[entity] = self.winners[entity] + 1 if entity in self.winners else 1
            
            
# Full regex matching and syntactic tree to find awards and all other entities with passed award object
class SemanticParseWithAward(Strategy):
    def __init__(self, award: Award_Category):
        super().__init__("SemanticParseWithAward")
        self.award = award
        self.presenters = {}
        self.nominees = {}
        self.winners = {}
        
    def extract(self, tweet: Dict):
        text = tweet['clean_text'].lower()
        winner_split, nominee_split, presenter_split = self.get_keyword_splits(text)
        splits = [presenter_split,nominee_split, winner_split]
        nominee_found = False
        for i in range(len(splits)):
            if nominee_found:
                break
            if i == 0:
                names = extract_name(tweet)
                if len(names) == 1:
                    self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[0] in self.presenters else 1
                else:
                    index = text.find(splits[i][1])
                    split_word = splits[i][1]
                    for name in names:
                        if split_word in self.leftwords and name in text[:index]:
                            self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[0] in self.presenters else 1
                        elif split_word in self.rightwords and name in text[index:]:
                            self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[0] in self.presenters else 1
                        elif split_word in self.bothwords:
                            self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[0] in self.presenters else 1
                continue
                    
            if self.award.entity_type == "person":
                entities = extract_name(tweet)
                
            elif self.award.entity_type == "movie":
                entities = extract_movie(tweet, self.award)
                if len(entities) == 1:
                    if i == 1:
                        self.nominees[entities[0]] = self.nominees[entities[0]] + 1 if entities[0] in self.nominees else 1
                        nominee_found = True
                    elif i == 2:
                        self.winners[entities[0]] = self.winners[entities[0]] + 1 if entities[0] in self.winners else 1
                else:
                    index = text.find(splits[i][1])
                    split_word = splits[i][1]
                    for entity in entities:
                        if entity in self.award.name:
                            continue
                        elif split_word in self.leftwords and entity in text[:index]:
                            if i == 0:
                                self.nominees[entity] = self.nominees[entity] + 1 if entity in self.nominees else 1
                                nominee_found = True

                            elif i == 1:
                                self.winners[entity] = self.winners[entity] + 1 if entity in self.winners else 1
                        elif split_word in self.rightwords and entity in text[index:]:
                            if i == 0:
                                self.nominees[entity] = self.nominees[entity] + 1 if entity in self.nominees else 1
                                nominee_found = True
                            elif i == 1:
                                self.winners[entity] = self.winners[entity] + 1 if entity in self.winners else 1
                        elif split_word in self.bothwords:
                            if i == 0:
                                self.nominees[entity] = self.nominees[entity] + 1 if entity in self.nominees else 1
                                nominee_found = True
                            elif i == 1:
                                self.winners[entity] = self.winners[entity] + 1 if entity in self.winners else 1
                            
        
class TimeBoxFreq(Strategy):
    def __init__(self, award: Award_Category):
        super().__init__("TimeBoxFreq")
        self.award = award
        award.affil_names_broad = award.affil_names
        award.affil_titles_broad = award.affil_titles
        award.hashtags_broad = award.hashtags
        
    def extract(self, data: List[Dict[str, Any]]):
        startIndex = self.award.startIndex
        endIndex = self.award.endIndex
        
        for i in range(len(startIndex)):
            for j in tqdm(range(startIndex[i], endIndex[i])):
                tweet = data[j]
                names = extract_name(tweet)
                if self.award.entity_type == "movie":
                    titles = extract_movie(tweet, self.award)
                # Add names to names affiliated with award
                for name in names:
                    if name in self.award.affil_names_broad:
                        self.award.affil_names_broad[name] += 1
                    else: 
                        self.award.affil_names_broad[name] = 1
                            
                # Add titles to titles affiliated with award
                if self.award.entity_type == "movie":
                    for title in titles:
                        if title in self.award.affil_titles_broad:
                            self.award.affil_titles_broad[title] += 1
                        else: 
                            self.award.affil_titles_broad[title] = 1
                            
                for hashtag in tweet["hashtags"]:
                    if hashtag in self.award.hashtags_broad:
                        self.award.hashtags_broad[hashtag] += 1
                    else:
                        self.award.hashtags_broad[hashtag] = 1
                
    
