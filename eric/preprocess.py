import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import ahocorasick
from multiprocessing import Pool
import pickle
from tqdm import tqdm
import nltk
import pandas as pd
from datetime import datetime, timezone
from operator import itemgetter
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from .utils import Award_Category, extract_name, extract_movie
from ftfy import fix_text
from unidecode import unidecode
from langdetect import detect_langs
from inflection import humanize, underscore


class Preprocessor:
    def __init__(self, name: str, requirements: Optional[List[str]] = None):
        self.name = name

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class WordsMatch(Preprocessor):
    def __init__(self, words=None, name: Optional[str] = None, remove: bool = True):
        super().__init__("WordsMatch" if name is None else name)
        if words is None:
            words = ["win", "won", "wins", "winner", 'get', 'got', 'gets', 'getting', 'gotten',
                     'take', 'took', 'takes', 'taken']
        self.words = words
        self.remove = remove

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for item in tqdm(data):
            tmp = []
            for word in self.words:
                pos = item['text'].lower().find(word)
                if pos != -1:
                    if pos != 0 and item['text'][pos - 1].isalpha():
                        continue
                    tmp.append((pos, word))
            if not tmp and self.remove:
                continue
            item[self.name] = tmp
            result.append(item)
        return result


class NLTK(Preprocessor):
    def __init__(self, name: Optional[str] = None, proc_num: int = 8, remove: bool = True):
        super().__init__("NLTK" if name is None else name)
        self.proc_num = proc_num
        self.remove = remove

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pool = Pool(self.proc_num)
        # split data into chunks
        data = [data[i:i + len(data) // self.proc_num] for i in range(0, len(data), len(data) // self.proc_num)]
        result = pool.map(self._process, data)
        pool.close()
        pool.join()
        result = [item for sublist in result for item in sublist]
        return result

    def _process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("Processing data with NLTK")
        result = []
        for item in tqdm(data):
            tokens = nltk.word_tokenize(item['text'])
            tmp = [i[0] for i in nltk.pos_tag(tokens) if i[1] == 'NNP']
            if not tmp and self.remove:
                continue
            item[self.name] = tmp
            result.append(item)
        return result


class AhoCorasickAutomaton(Preprocessor):
    def __init__(self, file_path: str, name: Optional[str] = None, remove: bool = True, year: int = 2013):
        self.path = Path(file_path)
        if self.path.exists():
            if self.path.suffix == ".pkl":
                print("Loading automaton from pickle file")
                self.automaton = pickle.load(open(self.path, "rb"))
            elif self.path.suffix == ".tsv":
                print("Building automaton from tsv file")
                self.automaton = ahocorasick.Automaton()
                data = pd.read_csv(self.path, sep='\t')
                print("Adding words to automaton")
                if 'primaryName' in data.columns:
                    for id, item in enumerate(tqdm(data['primaryName'])):
                        if not isinstance(item, str):
                            continue
                        if item.find(' ') == -1:
                            if data['birthYear'][id] == '\\N':
                                continue
                            if data['deathYear'][id] != '\\N' and int(data['deathYear'][id]) < year:
                                continue
                            if (str(data['primaryProfession'][id]).find('actress') == -1 
                                and str(data['primaryProfession'][id]).find('actor') == -1
                                and str(data['primaryProfession'][id]).find('director') == -1
                                and str(data['primaryProfession'][id]).find('producer') == -1
                                and str(data['primaryProfession'][id]).find('soundtrack') == -1
                                and str(data['primaryProfession'][id]).find('writer') == -1
                                and str(data['primaryProfession'][id]).find('composer') == -1):
                                continue
                        if len(item) < 4 or len(item.split(' ')) == 1:
                            # ignore short names
                            continue
                        # matching from the beginning of the word
                        self.automaton.add_word(f" {str(item).lower()}", item)
                else:
                    for id, item in enumerate(tqdm(data['primaryTitle'])):
                        if not isinstance(item, str):
                            continue
                        if not (data['titleType'][id] == 
                                'movie') and not (data['titleType'][id]==
                                                  'tvSeries') and not (data['titleType'][id]==
                                                                       'tvMiniSeries') and not(data['titleType'][id]==
                                                                                               'tvSeries') and not(data['titleType'][id]=='short'):
                            continue
                        if data['startYear'][id] == '\\N':
                            continue
                        if (int(data['startYear'][id]) <= year - 2 or int(data['startYear'][id]) > year):
                            if data['titleType'][id] == 'tvSeries' and int(data['startYear'][id]) <= year - 2:
                                if data['endYear'][id] != '\\N':
                                    continue
                            else:
                                continue
                        if len(item) < 3:
                            # ignore short names
                            continue
                        # matching from the beginning of the word
                        self.automaton.add_word(f" {str(item).lower()}", item)
                print("Making automaton")
                self.automaton.make_automaton()
                pickle.dump(self.automaton, open(self.path.with_suffix(".pkl"), "wb"))
                print(f"Automaton saved to pickle file:{self.path.with_suffix('.pkl')}")
            else:
                raise "File type not supported"
        self.remove = remove
        super().__init__("AhoCorasickAutomaton" if name is None else name)

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for item in tqdm(data):
            # using automaton.iter is faster
            tmp = []
            # add space to the beginning to match from the beginning of the word
            for each in self.automaton.iter(f" {item['text'].lower()}"):
                tmp.append(each)
            if not tmp and self.remove:
                continue
            item[self.name] = tmp
            result.append(item)
        return result
    
class AhoCorasickAutomaton(Preprocessor):
    def __init__(self, file_path: str, name: Optional[str] = None, remove: bool = True, year: int = 2013):
        self.path = Path(file_path)
        if self.path.exists():
            if self.path.suffix == ".pkl":
                print("Loading automaton from pickle file")
                self.automaton = pickle.load(open(self.path, "rb"))
            elif self.path.suffix == ".tsv":
                print("Building automaton from tsv file")
                self.automaton = ahocorasick.Automaton()
                data = pd.read_csv(self.path, sep='\t')
                print("Adding words to automaton")
                if 'primaryName' in data.columns:
                    self.automaton_type = "actor"
                    for id, item in enumerate(tqdm(data['primaryName'])):
                        if not isinstance(item, str):
                            continue
                        if item.find(' ') == -1:
                            if data['birthYear'][id] == '\\N':
                                continue
                            if data['deathYear'][id] != '\\N' and int(data['deathYear'][id]) < year:
                                continue
                            if (str(data['primaryProfession'][id]).find('actress') == -1 
                                and str(data['primaryProfession'][id]).find('actor') == -1
                                and str(data['primaryProfession'][id]).find('director') == -1
                                and str(data['primaryProfession'][id]).find('producer') == -1
                                and str(data['primaryProfession'][id]).find('soundtrack') == -1
                                and str(data['primaryProfession'][id]).find('writer') == -1
                                and str(data['primaryProfession'][id]).find('composer') == -1):
                                continue
                        if len(item) < 4 or len(item.split(' ')) == 1:
                            # ignore short names
                            continue
                        # matching from the beginning of the word
                        self.automaton.add_word(f" {str(item).lower()}", item)
                else:
                    self.automaton_type = "movie"
                    for id, item in enumerate(tqdm(data['primaryTitle'])):
                        if not isinstance(item, str):
                            continue
                        if not (data['titleType'][id] == 
                                'movie') and not (data['titleType'][id]==
                                                  'tvSeries') and not (data['titleType'][id]==
                                                                       'tvMiniSeries') and not(data['titleType'][id]==
                                                                                               'tvSeries') and not(data['titleType'][id]=='short'):
                            continue
                        if data['startYear'][id] == '\\N':
                            continue
                        if (int(data['startYear'][id]) <= year - 2 or int(data['startYear'][id]) > year):
                            if data['titleType'][id] == 'tvSeries' and int(data['startYear'][id]) <= year - 2:
                                if data['endYear'][id] != '\\N':
                                    continue
                            else:
                                continue
                        if len(item) < 3:
                            # ignore short names
                            continue
                        # matching from the beginning of the word
                        self.automaton.add_word(f" {str(item).lower()}", item)
                print("Making automaton")
                self.automaton.make_automaton()
                pickle.dump(self.automaton, open(self.path.with_suffix(".pkl"), "wb"))
                print(f"Automaton saved to pickle file:{self.path.with_suffix('.pkl')}")
            else:
                raise "File type not supported"
        self.remove = remove
        super().__init__("AhoCorasickAutomaton" if name is None else name)

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for item in tqdm(data):
            # using automaton.iter is faster
            tmp = []
            if self.automaton_type == "movie":
                hashtags = re.findall(r"#(\w+)", item['text'])
                for hashtag in hashtags:
                    if hashtag == "GoldenGlobes":
                        continue
                    else:
                        snake_name = hashtag[:] if "_" in hashtag else underscore(hashtag)
                        for match in self.automaton.iter(f" {humanize(snake_name).lower()}"):
                            tmp.append(match)
            
            if self.automaton_type == "actor":
                usernames = re.findall(r"@\S+", item['text'])
                for username in usernames:
                    snake_name = username[:] if "_" in username else underscore(username)
                    for match in self.automaton.iter(f" {humanize(username).lower()}"):
                        tmp.append(match)
                        
            # add space to the beginning to match from the beginning of the word
            for each in self.automaton.iter(f" {item['text'].lower()}"):
                tmp.append(each)
            if not tmp and self.remove:
                continue
            item[self.name] = tmp
            result.append(item)
        return result
    
class TimeSort(Preprocessor):
    def __init__(self, est_start_time: int, name: Optional[str] = None):
        super().__init__("TimeBox" if name is None else name)
        
    def process(self, data: List[Dict[str, Any]]) -> list[Dict[str, Any]]: 
        # First sort all the Tweets by timestamp
        print("Sorting data by timestamp")
        tqdm(data.sort(key=itemgetter('timestamp_ms')))
        return data


class FirstPass(Preprocessor):
    def __init__(self, est_start_time: int, name: Optional[str] = None):
        super().__init__("TimeBox" if name is None else name)
        
        
    def process(self, data: List[Dict[str, Any]]) -> list[Dict[str, Any]]:
        from .utils import Award_Category
        minute_indices = {}
        
        # calculate duration
        duration = datetime.fromtimestamp(data[-1]['timestamp_ms']/1000,tz = timezone.utc) - datetime.fromtimestamp(data[0]['timestamp_ms']/1000, tz = timezone.utc)
        minutes = round(duration.total_seconds()/60)
        
        # create a list of award names
        award_regexes = list(award for award in Award_Category.award_regex_dict.keys())
        award_regexes.append("total")
        
        # create a dictionary to store the number of tweets for each event over time
        freqs = {award_regex: np.zeros(minutes+1) for award_regex in award_regexes}
        
        # For each event, add count based on regex matches
        # TODO - implement many to one matching
        for i in tqdm(range(len(data))):
            tweet = data[i]
            curr_minute = round((datetime.fromtimestamp(tweet['timestamp_ms']/1000, tz = timezone.utc) - datetime.fromtimestamp(data[0]['timestamp_ms']/1000, tz = timezone.utc)).total_seconds()/60)

            if i>0:
                prev = data[i-1]
                if curr_minute != round((datetime.fromtimestamp(prev['timestamp_ms']/1000, tz = timezone.utc) - datetime.fromtimestamp(data[0]['timestamp_ms']/1000, tz = timezone.utc)).total_seconds()/60):
                    minute_indices[curr_minute] = i
                    
            freqs["total"][curr_minute] += 1
            if tweet['id'] == 290621553153036289:
                print(tweet['text'])
            for award_regex in Award_Category.award_regex_dict.keys():
                found = False
                if bool(re.search(award_regex, tweet['text'],re.IGNORECASE)):
                    found = True
                    names = []
                    titles = []
                    if not tweet["text"].startswith("RT"):
                        freqs[award_regex][curr_minute] += 1
                    names = extract_name(tweet)
                    if Award_Category.award_regex_dict[award_regex].entity_type == "movie":
                        titles = extract_movie(tweet, Award_Category.award_regex_dict[award_regex])
                    # Add names to names affiliated with award
                    for name in names:
                        if name in Award_Category.award_regex_dict[award_regex].affil_names:
                            Award_Category.award_regex_dict[award_regex].affil_names[name] += 1
                        else: 
                            Award_Category.award_regex_dict[award_regex].affil_names[name] = 1
                            
                    # Add titles to titles affiliated with award
                    for title in titles:
                        if title in Award_Category.award_regex_dict[award_regex].affil_titles:
                            Award_Category.award_regex_dict[award_regex].affil_titles[title] += 1
                        else: 
                            Award_Category.award_regex_dict[award_regex].affil_titles[title] = 1
                    # print(tweet['text'])
                    # print(curr_minute)
                    
                    for hashtag in tweet["hashtags"]:
                        if hashtag in Award_Category.award_regex_dict[award_regex].hashtags:
                            Award_Category.award_regex_dict[award_regex].hashtags[hashtag] += 1
                        else:
                            Award_Category.award_regex_dict[award_regex].hashtags[hashtag] = 1
                    
                    # Add index of tweet to award.tweet_indices
                    np.append(Award_Category.award_regex_dict[award_regex].tweet_indices, i)
                    # Add a field to the data that stores the award name
                    data[i]["award"] = Award_Category.award_regex_dict[award_regex].name
                    break
                
            if found == False:
                data[i]["award"] = "None"
        
        # Find the spike in the number of tweets for each event
        #print(freqs)
        stds = [np.std(value) for value in freqs.values()]
        
        window_size = 6
        moving_avgs = [np.convolve(value, np.ones(window_size)/window_size, mode = 'full')[window_size -1:] for value in freqs.values()]
        max_intervals = [np.argsort(-moving_avg)[:2] for moving_avg in moving_avgs]
        
        for i in range(len(max_intervals)-1):
            if abs(max_intervals[i][1]-max_intervals[i][0]) < window_size:
                Award_Category.award_regex_dict[award_regexes[i]].startIndex.append(minute_indices[max_intervals[i][0]])
                Award_Category.award_regex_dict[award_regexes[i]].endIndex.append(minute_indices[max_intervals[i][0]+window_size])
            else:
                Award_Category.award_regex_dict[award_regexes[i]].startIndex.append(minute_indices[max_intervals[i][0]])
                Award_Category.award_regex_dict[award_regexes[i]].endIndex.append(minute_indices[max_intervals[i][0]+window_size])
                Award_Category.award_regex_dict[award_regexes[i]].startIndex.append(minute_indices[max_intervals[i][1]])
                Award_Category.award_regex_dict[award_regexes[i]].endIndex.append(minute_indices[max_intervals[i][1]+window_size])
        print("done with first pass")
        # find the beginning of the ceremony
        # TODO - implement a more robust way to find the beginning of the ceremony
        
        return data

class PreprocessText(Preprocessor):
    def __init__(self, name: Optional[str] = None):
        super().__init__("StripText" if name is None else name)

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]: 
        for i in tqdm(range(len(data))):
            data[i]['text'] = fix_text(data[i]['text'])
            data[i]['text'] = unidecode(data[i]['text'])
            data[i]['text'] = " ".join(data[i]['text'].split())
            
            # Strip away urls, hashtags, and mentions
            data[i]["clean_text"] = data[i]['text']
            data[i]["clean_text"]= re.sub(r"http\S+", "", data[i]['clean_text'])
            hashtags = re.findall(r"#(\w+)", data[i]['clean_text'])
            data[i]['hashtags'] = []
            for hashtag in hashtags:
                data[i]['hashtags'].append(hashtag)
            data[i]["clean_text"]=re.sub(r"#(\w+)", "", data[i]['clean_text'])
            data[i]["clean_text"]=re.sub(r"@\S+", "", data[i]['clean_text'])
            data[i]["clean_text"]=re.sub(r"RT", "", data[i]['clean_text'])
            data[i]["clean_text"] = " ".join(data[i]['clean_text'].split())
        return data

class Summarize(Preprocessor):
    def __init__(self, name: Optional[str] = None, nltk_name: str = 'NLTK',
                 acautomaton_name: str = 'AhoCorasickAutomaton',
                 remove: bool = True):
        super().__init__("Summarize" if name is None else name)
        self.nltk_name = nltk_name
        self.acautomaton_name = acautomaton_name
        self.remove = remove

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for item in tqdm(data):
            tmp = []
            for i in item[self.nltk_name]:
                for j in item[self.acautomaton_name]:
                    if i in j[1]:
                        tmp.append(j)
            if not tmp and self.remove:
                continue
            tmp = list(set(tmp))
            item[self.name] = []
            # remove part of names
            for idi, i in enumerate(tmp):
                flag = False
                for idj, j in enumerate(tmp):
                    if idi == idj:  # fix a bug: must use idi==idj, not i == j
                        continue
                    if (j[1].startswith(i[1]) or j[1].endswith(i[1])) and i[1] != j[1]:
                        flag = True
                        break
                if not flag:
                    item[self.name].append(i)
            if not item[self.name]:
                item[self.name] = item[self.acautomaton_name]
            result.append(item)
        return result


class Duplicate(Preprocessor):
    def __init__(self, name: Optional[str] = None):
        super().__init__("Duplicate" if name is None else name)

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ans = dict()
        for i in data:
            ans[i['text']] = i
        return list(ans.values())


class ReMatch(Preprocessor):
    def __init__(self, name: Optional[str] = None, remove: bool = True, exps=None):
        super().__init__("ReMatch" if name is None else name)
        if exps is None:
            exps = []
        self.exps = exps
        self.remove = remove

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for item in tqdm(data):
            tmp = []
            for exp in self.exps:
                tmp += re.findall(exp, item['text'])
            if not tmp and self.remove:
                continue
            item[self.name] = tmp
            result.append(item)
            # print(item['text'])
        return result


class Sentiment(NLTK):
    def __init__(self, name: Optional[str] = None, proc_num: int = 8):
        super().__init__("Sentiment" if name is None else name, proc_num=proc_num, remove=False)

    def _process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("Processing data with NLTK")
        analyzer = SentimentIntensityAnalyzer
        result = []
        for item in tqdm(data):
            score = analyzer.polarity_scores(item['text'])
            item[self.name] = score
            result.append(item)
        return result


class PreprocessPipe:
    def __init__(self):
        self.preprocessors = []

    def add_processor(self, processor: Preprocessor):
        self.preprocessors.append(processor)

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for processor in self.preprocessors:
            l = len(data)
            data = processor.process(data)
            print(f"Processor {processor.name} processed {l - len(data)} items")
        print(f"Total data number now: {len(data)}")
        return data
