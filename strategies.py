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

def lookLeft(text, )

class Strategy:
    def __init__(self, name: str, requirements: Optional[List[str]] = None):
        self.name = name
        self.winner_regex = r"(won|wins|winner|goes to|acceptance speech|congrats|congratulations|congratulate)"
        self.nominee_regex = r"(nominations|nominee|nominated|rob|robbing|robbed|should\'ve)"
        self.presenter_regex = r"(present|presented|presents|presenting)"
        self.leftwords = []
        self.rightwords = []

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
    def __init__(self, name:str, timebox_award: Optional[Award_Category]=None):
        super().__init__("KeywordRegex" if name is None else name)
        self.timebox_award = timebox_award
        self.presenters = {}
        self.nominees = {}
        self.winners = {}
        
    def extract(self, tweet: Dict):
        text = tweet['clean_text'].lower()
        winner_split, nominee_split, presenter_split = self.get_keyword_splits(text)
        splits = [presenter_split, nominee_split, winner_split]
            
            
# Full regex matching and syntactic tree to find awards and all other entities with passed award object
class SemanticParseWithAward(Strategy):
    def __init__(self, name:str, award: Award_Category):
        super().__init__("KeywordRegex" if name is None else name)
        self.award = award
        self.presenters = {}
        self.nominees = {}
        self.winners = {}
        
    def extract(self, tweet: Dict):
        text = tweet['clean_text'].lower()
        winner_split, nominee_split, presenter_split = self.get_keyword_splits(text)
        splits = [nominee_split, winner_split, presenter_split]
        for i in range(len(splits)):
            if self.award.entity_type == "person":
                names = extract_name(tweet)
                if len(names) == 1:
                    if i == 0:
                        self.nominees[names[0]] = self.nominees[names[0]] + 1 if names[0] in self.nominees else 1
                    elif i == 1:
                        self.winners[names[0]] = self.winners[names[0]] + 1 if names[0] in self.winners else 1
                    elif i == 2:
                        self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[0] in self.presenters else 1
            elif self.award.entity_type == "movie":
                titles = extract_movie(tweet, self.award)
                if len(titles) == 1:
                    if i == 0:
                        self.nominees[titles[0]] = self.nominees[titles[0]] + 1 if titles[0] in self.nominees else 1
                    elif i == 1:
                        self.winners[titles[0]] = self.winners[titles[0]] + 1 if titles[0] in self.winners else 1
                    elif i == 2:
                        self.presenters[titles[0]] = self.presenters[titles[0]] + 1 if titles[0] in self.presenters else 1
                
        
class TimeBoxFreq(Strategy):
    def __init__(self, name:str, award: Award_Category):
        super().__init__("KeywordRegex" if name is None else name)
        self.award = award
        award.affil_names_broad = award.affil_names
        award.affil_titles_broad = award.affil_titles
        award.hashtags_broad = award.hashtags
        self.candidates = {}
        
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
                


