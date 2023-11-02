import regex as re
import spacy
from spacy import displacy
from collections import Counter
import nltk
import numpy as np


nlp = spacy.load("en_core_web_sm")




# award_regex_dict = {
    
#     r'best\s+screenplay': "best screenplay - motion picture",
#     r'best\s+director' : "best director - motion picture",
#     r'(?=.*comedy\s+or\s+musical)best\s+(?:performance\s+by\s+an\s+)?actress\s+(?:in\s+)?(?:a\s+)?(?:television|tv)\s+series': "best performance by an actress in a television series - comedy or musical",
#     r'foreign\s+language\s+film' : "best foreign language film",
    
#     r'(?=.*drama)best\s+(?:performance\s+by\s+an\s+)?actor\s+(?:in\s+)?(?:a\s+)?(?:television|tv)\s+series' : "best performance by an actor in a television series - drama",
    
# }


def extract_name(tweet):
  one_word_names = ["madonna", "zendaya", "adele", "charo", "teller", "tiffany", "banksy", "lalaine", "iman", "prince", "shakira", "cheryl"]
  if len(tweet["name"]) == 0:
    return []
  else: 
    names = [name[1] for name in tweet["name"] if len(re.findall(r'\w+', name[1])) > 1 and name[1] not in one_word_names and not bool(re.search(r'\b' + re.escape("the") + r'\b',name[1],re.IGNORECASE))] 
    names_merge = names.copy()
    for s1 in names:
        for s2 in names:
            merged = False
            if s1 != s2 and s1 in s2 and s1 in names_merge and s2 in names_merge:
                # Merge s1 into s2 and remove s1 from merged_list
                names_merge.remove(s1)
                names_merge.remove(s2)
                names_merge.append(s2.replace(s1, s1 + " " + s2))
                merged = True
            if merged:
                break

    # Remove duplicates from the merged list
    return list(set(names_merge))
    
def extract_movie(tweet, award):
  if len(tweet["movie"]) == 0:
    return []
  else: 
    movies = [movie[1] for movie in tweet["movie"] if not bool(re.search(r'\b' + re.escape(movie[1]) + r'\b',award.name,re.IGNORECASE))]
    movie_merge = movies.copy()
    for s1 in movies:
        for s2 in movies:
            merged = False
            if s1 != s2 and s1 in s2 and s1 in movie_merge and s2 in movie_merge:
                # Merge s1 into s2 and remove s1 from merged_list
                movie_merge.remove(s1)
                movie_merge.remove(s2)
                movie_merge.append(s2.replace(s1, s1 + " " + s2))
                merged = True
            if merged:
                break

    # Remove duplicates from the merged list
    movies = list(set(movie_merge))
    for movie in movies:
        for name in tweet["name"]:
            if movie in name[1] and movie in movies:
                movies.remove(movie)
                
    return movies

class Award_Category:
    award_regex_dict = {}
    person_words = ["role", "director", "actor", "actress", "directing"]

    
    def __init__(self, name: str):
        self.name = name.lower()
        self.entity_type = self.detectEntityTypeFromAwardName()
        self.regex_string = self.regexStringFromAwardName()
        Award_Category.award_regex_dict[self.regex_string] = self
        self.nominees = []
        self.presenters = []
        self.winner = None
        
        self.affil_names = {}
        self.affil_titles = {}
        self.affil_names_broad = {}
        self.affil_titles_broad = {}
        self.hashtags = {}
        self.hashtags_broad = {}
        
        self.startIndex = []
        self.endIndex = []
        self.tweet_indices = []
        self.potentialNominees = {}
        self.potentialPresenters = {}
        self.potentialWinner = {}
        self.simpleFrameCandidates = {"winner": {}, "nominees": {}, "presenters": {}}
        
    @classmethod
    def sortRegexDict(cls):
        cls.award_regex_dict = dict(sorted(cls.award_regex_dict.items(), key=lambda item: len(item[0]), reverse=True))
    
    def detectEntityTypeFromAwardName(self):
        # Other types of names other than "best ___"
        if not bool(re.search(r'\b' + re.escape("best") + r'\b',self.name,re.IGNORECASE)):
            return "person"
        else:
            for word in Award_Category.person_words:
                pattern = r'\b' + re.escape(word) + r'\b'
                if bool(re.search(pattern, self.name, re.IGNORECASE)):
                    return "person"
            if bool(re.search(r'\b' + re.escape("song") + r'\b',self.name,re.IGNORECASE)):
                return "movie"
            else:
                return "movie"

    def regexStringFromAwardName(self):
        # make a case for keyword
        # television
        # or 
        # award
        regex_string = r''
        optionals = ["performance", "motion", "series", "picture", "made", "role","mini-series", "feature"]
        words = nltk.word_tokenize(self.name)
        tagged = nltk.pos_tag(words)
        if bool(re.search(r'\b' + re.escape("award") + r'\b',self.name,re.IGNORECASE)):
            regex_string += re.escape(self.name.split("award")[0])
            return regex_string
            
        skip = 0
        for i in range(0, len(words)):
            # keywords
            if skip > 0:
                skip -= 1
                continue
            if tagged[i][0] in optionals or tagged[i][1] == ":" or tagged[i][1] == "IN" or tagged[i][1] == "CC" or tagged[i][1] == "DT" or tagged[i][0] == ",":
                continue
            elif tagged[i][0] == ("television" or "tv"):
                regex_string += r'(?=.*television|tv)'
                continue
            elif i < len(words) -2 and tagged[i+1][0] == ("or"):
                if i < len(words) - 3 and tagged[i+2] == "motion" and tagged[i+3] == "picture":
                  regex_string += r'(?=.*'+re.escape(tagged[i][0]) + r'|' + re.escape("motion picture") + r')'
                  skip = 3
                else:
                  regex_string += r'(?=.*'+re.escape(tagged[i][0]) + r'|' + re.escape(tagged[i+2][0]) + r')'
                  skip = 2
                continue
            else:
                regex_string += r'(?=.*' + re.escape(tagged[i][0]) + r')'
                continue
            
        return regex_string


