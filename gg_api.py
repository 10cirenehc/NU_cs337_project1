'''Version 0.4'''
from typing import List, Dict, Any
import re

from tqdm import tqdm

from award_filter import Award
from preprocess import PreprocessPipe, AhoCorasickAutomaton, NLTK, WordsMatch, Summarize, Duplicate, TimeSort, PreprocessText
from strategies import FirstPass
from utils import Award_Category
import json

def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''
    data = json.load(open("data/gg2013.json", "r"))
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    pipe.add_processor(WordsMatch(words=['host']))
    pipe.add_processor(AhoCorasickAutomaton("data/actors.pkl"))
    pipe.add_processor(NLTK(proc_num=12))
    pipe.add_processor(Summarize())
    data = pipe.process(data)
    hosts = dict()
    for i in data:
        for name in i['Summarize']:
            hosts[name[1]] = hosts[name[1]] + 1 if name[1] in hosts else 1

    host = sorted(hosts.items(), key=lambda x: x[1], reverse=True)
    return [i[0] for i in host[:2]]

def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''
    from award_filter import get_award_name
    return get_award_name()

def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''
    # Your code here
    return nominees

def get_winner(year):
    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    return winners

def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    # Your code here
    return presenters

def pre_ceremony():
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    
    print("Pre-ceremony processing complete.")
    return

def main():
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''
    # data = json.load(open("data/gg2013.json", "r"))
    coded_categories = list(json.load(open("data/gg2013answers.json","r"))["award_data"].keys())
    for category in tqdm(coded_categories):
        a = Award_Category(category)
    Award_Category.sortRegexDict()
    print(Award_Category.award_regex_dict)
    # sorter = TimeSort(5)
    # data = sorter.process(data)
    # json.dump(data, open("data/gg2013_sorted.json", "w"), indent=4)
    data = json.load(open("data/gg2013_sorted.json", "r"))
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    pipe.add_processor(AhoCorasickAutomaton("data/actors.tsv", remove=False))
    pipe.add_processor(AhoCorasickAutomaton("data/movie.tsv", remove=False, name='movie'))
    pipe.add_processor(NLTK(proc_num=12, remove=False))
    pipe.add_processor(Summarize(remove=False, name="name"))
    data = pipe.process(data)
    json.dump(data, open("data/gg2013_sorted_annotated.json", "w"), indent=4)

    # data = json.load(open("data/gg2013_sorted_annotated.json", "r"))
    textPreprocess = PreprocessText()
    textPreprocess.process(data)
    json.dump(data, open("data/gg2013_sorted_annotated.json", "w"), indent=4)
    data = json.load(open("data/gg2013_sorted_annotated.json", "r"))
    firstPass = FirstPass(5)
    #print(timeBox.process(data))
    data = firstPass.extract(data)
    return

if __name__ == '__main__':
    main()
    #get_awards(2013)