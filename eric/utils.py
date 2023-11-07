import regex as re
import spacy
from spacy import displacy
from collections import Counter
import nltk
import numpy as np
from datetime import datetime, timezone

nlp = spacy.load("en_core_web_sm")
from spacy.matcher import Matcher


# award_regex_dict = {

#     r'best\s+screenplay': "best screenplay - motion picture",
#     r'best\s+director' : "best director - motion picture",
#     r'(?=.*comedy\s+or\s+musical)best\s+(?:performance\s+by\s+an\s+)?actress\s+(?:in\s+)?(?:a\s+)?(?:television|tv)\s+series': "best performance by an actress in a television series - comedy or musical",
#     r'foreign\s+language\s+film' : "best foreign language film",

#     r'(?=.*drama)best\s+(?:performance\s+by\s+an\s+)?actor\s+(?:in\s+)?(?:a\s+)?(?:television|tv)\s+series' : "best performance by an actor in a television series - drama",

# }

def normalize_dict(dict, total, n):
    for key in dict:
        dict[key] = dict[key] / total
    if len(dict) < n:
        n = -1
    return {key: dict[key] for key in list(dict)[:n]}


def substring_cluster(dict):
    # take dictionary of string keys with number values as values and cluster the keys that are substrings of each other
    # return a dictionary of the same format
    # dict is already sorted by value
    dict2 = dict.copy()
    for s1 in dict:
        for s2 in dict:
            merged = False
            if s1 != s2 and s1 in s2 and s1 in dict and s2 in dict:
                # Merge s1 into s2 and remove s1 from merged_list
                dict[s2] += dict2[s1]
                dict2.pop(s1)
                merged = True
            if merged:
                break
    return dict2


def sort_dict_by_value(dict):
    return {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}


def extract_name(tweet):
    one_word_names = ["madonna", "zendaya", "adele", "charo", "teller", "tiffany", "banksy", "lalaine", "iman",
                      "prince", "shakira", "cheryl"]
    if len(tweet["name"]) == 0:
        return []
    else:
        names = [name[1] for name in tweet["name"] if
                 len(re.findall(r'\w+', name[1])) > 1 and name[1] not in one_word_names and
                 not bool(re.search(r'\b' + re.escape("the") + r'\b', name[1], re.IGNORECASE))]
        for name in names:
            if (name.lower() == "an actor" or name.lower() == "an actress" or name.lower().startswith("director")) and name in names:
                names.remove(name)
                
        names_merge = names.copy()
        for s1 in names:
            for s2 in names:
                merged = False
                if s1 != s2 and s1 in s2 and s1 in names_merge and s2 in names_merge:
                    # Merge s1 into s2 and remove s1 from merged_list
                    names_merge.remove(s1)
                    merged = True
                if merged:
                    break

        # Remove duplicates from the merged list
        return list(set(names_merge))


def extract_movie(tweet, award):
    # Do a lot of work on this
    ignore = ["gold", "golden", "globe", "oscar", "oscars", "wins", "win", "won", "nominated", "present", "congrats",
              "congratulations", "robbing", "goes to",
              "winner", "acceptance", "speech", "congratulate", "should've", "robbed", "rob", "nominations", "nominee",
              "presented", "presents", "drama", "play", "and",
              "give", "still", "good", "did", "didn't", "you", "pre", "just", "wait", "can", "show", "so far", "was",
              "winners", "like", "san", "los", 'con', "not", "her", "tho", "omg", "fuck", "got"
                                                                                          "thin", "because", "one",
              "how", "made", "happy", "better", "motion picture", "watching", "even", "more", "over", "looking", "look",
              "lol", "don't", "pic",
              "man", "that", "red carpet", "tonight", "they", "got", "gol", "best actor", "best actress", "the best"]
    year = datetime.fromtimestamp(tweet['timestamp_ms'] / 1000, tz=timezone.utc).year
    ignore.append(str(year))
    if len(tweet["movie"]) == 0:
        return []
    else:
        movies = [movie[1] for movie in tweet["movie"] if
                  not bool(re.search(re.escape(movie[1]), award.name, re.IGNORECASE)) and not movie[
                                                                                                  1].lower() in ignore]

        # for movie in movies:
        #     if len(movie.split(" ")) == 1:
        #         matcher = Matcher(nlp.vocab)
        #         pattern = [{"TEXT": movie}]

        #         matcher.add("MOVIE", [pattern])
        #         doc = nlp(tweet["text"])

        #         matches = matcher(doc)
        #         for match_id, start, end in matches:
        #             if doc[start].pos_ != "PROPN" and doc[start].pos_ != "NOUN" and doc[start].dep_ != "NSUBJ" and doc[start].dep_ != "DOBJ" and doc[start].dep_ != "POBJ" and movie in movies:
        #                 movies.remove(movie)
        #                 break
        for movie in movies:
            if len(movie.split(" ")) == 1:
                if not movie.lower() in tweet["text"].lower().split(" ") and movie in movies:
                    movies.remove(movie)
            else:
                for word in movie.split(" "):
                    if not word.lower() in tweet["text"].lower().split(" ") and movie in movies:
                        movies.remove(movie)
                        break

        movie_merge = movies.copy()
        for s1 in movies:
            s1 = s1.lower()
            for s2 in movies:
                s2 = s2.lower()
                merged = False
                if s1 != s2 and s1 in s2 and s1 in movie_merge and s2 in movie_merge:
                    # Merge s1 into s2 and remove s1 from merged_list
                    movie_merge.remove(s1)
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

        self.name_co_occurance = {}

        self.startIndex = []
        self.endIndex = []
        self.tweet_indices = []
        self.potentialNominees = {}
        self.potentialPresenters = {}
        self.potentialWinners = {}
        self.simpleFrameCandidates = {"winner": {}, "nominees": {}, "presenters": {}}

    @classmethod
    def sortRegexDict(cls):
        cls.award_regex_dict = dict(sorted(cls.award_regex_dict.items(), key=lambda item: len(item[0]), reverse=True))

    def detectEntityTypeFromAwardName(self):
        # Other types of names other than "best ___"
        if not bool(re.search(r'\b' + re.escape("best") + r'\b', self.name, re.IGNORECASE)):
            return "person"
        else:
            for word in Award_Category.person_words:
                pattern = r'\b' + re.escape(word) + r'\b'
                if bool(re.search(pattern, self.name, re.IGNORECASE)):
                    return "person"
            if bool(re.search(r'\b' + re.escape("song") + r'\b', self.name, re.IGNORECASE)):
                return "movie"
            else:
                return "movie"

    def regexStringFromAwardName(self):
        # make a case for keyword
        # television
        # or
        # award
        regex_string = r''
        optionals = ["performance", "motion", "series", "picture", "made", "role", "mini-series", "feature"]
        words = nltk.word_tokenize(self.name)
        tagged = nltk.pos_tag(words)
        if bool(re.search(r'\b' + re.escape("award") + r'\b', self.name, re.IGNORECASE)):
            regex_string += re.escape(self.name.split("award")[0])
            return regex_string

        skip = 0
        for i in range(0, len(words)):
            # keywords
            if skip > 0:
                skip -= 1
                continue
            if tagged[i][0] in optionals or tagged[i][1] == ":" or tagged[i][1] == "IN" or tagged[i][1] == "CC" or \
                    tagged[i][1] == "DT" or tagged[i][0] == ",":
                continue
            elif tagged[i][0] == ("television" or "tv"):
                regex_string += r'(?=.*television|.*tv|.*series)'
                continue
            elif i < len(words) - 2 and tagged[i + 1][0] == ("or"):
                if i < len(words) - 3 and tagged[i + 2] == "motion" and tagged[i + 3] == "picture":
                    regex_string += r'(?=.*' + re.escape(tagged[i][0]) + r'|' + re.escape("motion picture") + r')'
                    skip = 3
                else:
                    regex_string += r'(?=.*' + re.escape(tagged[i][0]) + r'|' + re.escape(tagged[i + 2][0]) + r')'
                    skip = 2
                continue
            else:
                regex_string += r'(?=.*' + re.escape(tagged[i][0]) + r')'
                continue

        return regex_string


