import tqdm
from typing import List, Dict, Any
import re

from tqdm import tqdm
from strategies import * 
from award_filter import Award
from preprocess import PreprocessPipe, AhoCorasickAutomaton, NLTK, WordsMatch, Summarize, Duplicate, TimeSort, PreprocessText, DeDuplicate
from utils import Award_Category, sort_dict_by_value, substring_cluster, normalize_dict
import json

def get_award_data():
    coded_categories = list(json.load(open("data/gg2013answers.json","r"))["award_data"].keys())
    for category in tqdm(coded_categories):
        a = Award_Category(category)
    Award_Category.sortRegexDict()
    

    data = json.load(open("data/gg2013_sorted_annotated.json", "r"))
    deDup = DeDuplicate()
    data = deDup.process(data)
    textPreprocess = PreprocessText()
    data = textPreprocess.process(data)
    firstPass = FirstPass()
    #print(timeBox.process(data))
    data = firstPass.extract(data)
    
    winners = []
    
    for award in tqdm(Award_Category.award_regex_dict.values()):
        timeBoxFreq = TimeBoxFreq(award)
        timeBoxFreq.extract(data)
        
        
        timeBoxCoOccurance = TimeBoxCoOccurance(award)
        timeBoxCoOccurance.extract(data)
        
        # Extract the candidate matches for exact award matches
        exactAwardMatch = SemanticParseWithAward(award)
        for index in tqdm(award.tweet_indices):
            exactAwardMatch.extract(data[index])
        # Extract candidate matches for timebox tweets
        timeBoxAwardMatch = SemanticParseWithoutAward(award)
        indices = list(range(award.startIndex[0], award.endIndex[0]+1))
        for index in indices:
            timeBoxAwardMatch.extract(data[index])
        
        # cluster results
        award.affil_names = sort_dict_by_value(award.affil_names)
        award.affil_titles = sort_dict_by_value(award.affil_titles)
        award.affil_names_broad = sort_dict_by_value(award.affil_names_broad)
        award.affil_titles_broad = sort_dict_by_value(award.affil_titles_broad)
        # award.name_co_occurance = sort_dict_by_value(award.name_co_occurance)
        exactAwardMatch.nominees = sort_dict_by_value(exactAwardMatch.nominees)
        exactAwardMatch.presenters = sort_dict_by_value(exactAwardMatch.presenters)
        exactAwardMatch.winners = sort_dict_by_value(exactAwardMatch.winners)
        timeBoxAwardMatch.nominees = sort_dict_by_value(timeBoxAwardMatch.nominees)
        timeBoxAwardMatch.presenters = sort_dict_by_value(timeBoxAwardMatch.presenters)
        timeBoxAwardMatch.winners = sort_dict_by_value(timeBoxAwardMatch.winners)
        
        award.affil_names = substring_cluster(award.affil_names)
        award.affil_titles = substring_cluster(award.affil_titles)
        award.affil_names_broad = substring_cluster(award.affil_names_broad)
        award.affil_titles_broad = substring_cluster(award.affil_titles_broad)
        # award.name_co_occurance = substring_cluster(award.name_co_occurance)
        exactAwardMatch.nominees = substring_cluster(exactAwardMatch.nominees)
        exactAwardMatch.presenters = substring_cluster(exactAwardMatch.presenters)
        exactAwardMatch.winners = substring_cluster(exactAwardMatch.winners)
        timeBoxAwardMatch.nominees = substring_cluster(timeBoxAwardMatch.nominees)
        timeBoxAwardMatch.presenters = substring_cluster(timeBoxAwardMatch.presenters)
        timeBoxAwardMatch.winners = substring_cluster(timeBoxAwardMatch.winners)
        
        timeBoxLen = award.endIndex[0] - award.startIndex[0] + 1
        exactLen = len(award.tweet_indices)
        
        award.affil_names = normalize_dict(award.affil_names, exactLen, 10)
        award.affil_titles = normalize_dict(award.affil_titles, exactLen, 15)
        award.affil_names_broad = normalize_dict(award.affil_names_broad, timeBoxLen, 20)
        award.affil_titles_broad = normalize_dict(award.affil_titles_broad, timeBoxLen, 20)
        exactAwardMatch.nominees = normalize_dict(exactAwardMatch.nominees, exactLen, -1)
        exactAwardMatch.presenters = normalize_dict(exactAwardMatch.presenters, exactLen, -1)
        exactAwardMatch.winners = normalize_dict(exactAwardMatch.winners, exactLen, -1)
        timeBoxAwardMatch.nominees = normalize_dict(timeBoxAwardMatch.nominees, timeBoxLen, 7)
        timeBoxAwardMatch.presenters = normalize_dict(timeBoxAwardMatch.presenters, timeBoxLen, 5)
        timeBoxAwardMatch.winners = normalize_dict(timeBoxAwardMatch.winners, timeBoxLen, 5)
        
        # Extract the winner
        for name in exactAwardMatch.winners:
            weight = 6
            if name in award.potentialWinners:
                award.potentialWinners[name] += weight*exactAwardMatch.winners[name]
            else: 
                award.potentialWinners[name] = weight*exactAwardMatch.winners[name]
        
        for name in timeBoxAwardMatch.winners:
            weight = 10
            if name in award.potentialWinners:
                award.potentialWinners[name] += weight*timeBoxAwardMatch.winners[name]
            else: 
                award.potentialWinners[name] = weight*timeBoxAwardMatch.winners[name]
        
        if award.entity_type == "person":
            for name in award.affil_names:
                weight = 0.7
                if name in award.potentialWinners:
                    award.potentialWinners[name] += weight*award.affil_names[name]
                else: 
                    award.potentialWinners[name] = weight*award.affil_names[name]
                    
            for name in award.affil_names_broad:
                weight = 1
                if name in award.potentialWinners:
                    award.potentialWinners[name] += weight*award.affil_names_broad[name]
                else: 
                    award.potentialWinners[name] = weight*award.affil_names_broad[name]
                    
        if award.entity_type == "title":
            for name in award.affil_titles:
                weight = 0.7
                if name in award.potentialWinners:
                    award.potentialWinners[name] += weight*award.affil_titles[name]
                else: 
                    award.potentialWinners[name] = weight*award.affil_titles[name]
                    
            for name in award.affil_titles_broad:
                weight = 1
                if name in award.potentialWinners:
                    award.potentialWinners[name] += weight*award.affil_titles_broad[name]
                else: 
                    award.potentialWinners[name] = weight*award.affil_titles_broad[name]
                
        award.winner = list(sort_dict_by_value(award.potentialWinners).keys())[0]
        winners.append(award.winner)
        print("yay")
        
        award.timeBoxAwardMatch = timeBoxAwardMatch
        award.exactAwardMatch = exactAwardMatch
        
        

    for award in tqdm(Award_Category.award_regex_dict.values()):
        
        # Find the presenters
        for name in award.exactAwardMatch.presenters:
            if name == award.winner:
                continue
            weight = 10
            if name in award.potentialPresenters:
                award.potentialPresenters[name] += weight*award.exactAwardMatch.presenters[name]
            else: 
                award.potentialPresenters[name] = weight*award.exactAwardMatch.presenters[name]
                
        for name in award.timeBoxAwardMatch.presenters:
            if name in winners:
                continue
            weight = 15
            if name in award.potentialPresenters:
                award.potentialPresenters[name] += weight*award.timeBoxAwardMatch.presenters[name]
            else: 
                award.potentialPresenters[name] = weight*award.timeBoxAwardMatch.presenters[name]
        
        for name in award.affil_names:
            if name == award.winner or name in award.timeBoxAwardMatch.nominees:
                continue
            weight = 0.7
            if name in award.potentialPresenters:
                award.potentialPresenters[name] += weight*award.affil_names[name]
            else: 
                award.potentialPresenters[name] = weight*award.affil_names[name]
                
        for name in award.affil_names_broad:
            if name == award.winner or name in award.timeBoxAwardMatch.nominees:
                continue
            weight = 1
            if name in award.potentialPresenters:
                award.potentialPresenters[name] += weight*award.affil_names_broad[name]
            else: 
                award.potentialPresenters[name] = weight*award.affil_names_broad[name]        
        
        # Look up co_occurances for the top presenter 
        weight = 40
        for person in award.name_co_occurance.keys():
            for co_person in award.name_co_occurance[person]:
                if person == award.winner or award.name_co_occurance[person] == award.winner:
                    continue
                if person in award.potentialPresenters and co_person in award.potentialPresenters:
                    award.potentialPresenters[person] += award.name_co_occurance[person][co_person]/(timeBoxLen*2)
                    award.potentialPresenters[co_person] += award.name_co_occurance[person][co_person]/(timeBoxLen*2)

        award.potentialPresenters = sort_dict_by_value(award.potentialPresenters)
        award.presenters = list(sort_dict_by_value(award.potentialPresenters).keys())[0:2]
                    
        # Find the nominees
        for name in award.exactAwardMatch.nominees:
            if name in winners or name in award.presenters:
                continue
            weight = 3
            if name in award.potentialNominees:
                award.potentialNominees[name] += weight*award.exactAwardMatch.nominees[name]
            else: 
                award.potentialNominees[name] = weight*award.exactAwardMatch.nominees[name]     
                
        for name in award.timeBoxAwardMatch.nominees:
            if name in winners or name in award.presenters:
                continue
            weight = 15
            if name in award.potentialNominees:
                award.potentialNominees[name] += weight*award.timeBoxAwardMatch.nominees[name]
            else: 
                award.potentialNominees[name] = weight*award.timeBoxAwardMatch.nominees[name]
                
        if award.entity_type == "person":
            for name in award.affil_names:
                if name in winners or name in award.presenters:
                    continue
                weight = 0.7
                if name in award.potentialNominees:
                    award.potentialNominees[name] += weight*award.affil_names[name]
                else: 
                    award.potentialNominees[name] = weight*award.affil_names[name]
                    
            for name in award.affil_names_broad:
                if name in winners or name in award.presenters:
                    continue
                weight = 1
                if name in award.potentialNominees:
                    award.potentialNominees[name] += weight*award.affil_names_broad[name]
                else: 
                    award.potentialNominees[name] = weight*award.affil_names_broad[name]
                    
        if award.entity_type == "movie":
            for name in award.affil_titles:
                if name in winners or name in award.presenters:
                    continue
                weight = 0.7
                if name in award.potentialNominees:
                    award.potentialNominees[name] += weight*award.affil_titles[name]
                else: 
                    award.potentialNominees[name] = weight*award.affil_titles[name]
                    
            for name in award.affil_titles_broad:
                if name in winners or name in award.presenters:
                    continue
                weight = 1
                if name in award.potentialNominees:
                    award.potentialNominees[name] += weight*award.affil_titles_broad[name]
                else: 
                    award.potentialNominees[name] = weight*award.affil_titles_broad[name]  
                    
        award.potentialNominees = sort_dict_by_value(award.potentialNominees)
        award.nominees = list(sort_dict_by_value(award.potentialNominees).keys())[0:4]
       
    print("done")
        
if __name__ == "__main__":
    get_award_data()
        
    
    
    
    