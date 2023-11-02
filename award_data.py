import tqdm
from typing import List, Dict, Any
import re

from tqdm import tqdm
from strategies import * 
from award_filter import Award
from preprocess import PreprocessPipe, AhoCorasickAutomaton, NLTK, WordsMatch, Summarize, Duplicate, FirstPass, TimeSort, PreprocessText
from utils import Award_Category
import json

def get_award_data():
    coded_categories = list(json.load(open("data/gg2013answers.json","r"))["award_data"].keys())
    for category in tqdm(coded_categories):
        a = Award_Category(category)
    Award_Category.sortRegexDict()
    
    data = json.load(open("data/gg2013_sorted_annotated.json", "r"))
    
    firstPass = FirstPass(5)
    #print(timeBox.process(data))
    data = firstPass.process(data)
    
    for award in tqdm(Award_Category.award_regex_dict.values()):
        timeBoxFreq = TimeBoxFreq(award)
        timeBoxFreq.extract(data)
        
        # Extract the candidate matches for exact award matches
        exactAwardMatch = SemanticParseWithAward(award)
        for index in tqdm(award.tweet_indices):
            exactAwardMatch.extract(data[index])
        # Extract candidate matches for timebox tweets
        timeBoxAwardMatch = SemanticParseWithoutAward(award)
        indices = list(range(award.startIndex[0], award.endIndex[0]+1))
        for index in indices:
            timeBoxAwardMatch.extract(data[index])
            
    print("done")
        
if __name__ == "__main__":
    get_award_data()
        
    
    
    
    