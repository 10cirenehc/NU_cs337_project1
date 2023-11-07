'''Version 0.4'''
import re
from copy import deepcopy

from tqdm import tqdm

from award_filter import Award
from .preprocess import PreprocessPipe, AhoCorasickAutomaton, NLTK, WordsMatch, Summarize, Duplicate, TimeSort, \
    PreprocessText, ReMatch
from .strategies import FirstPass
from .utils import Award_Category
import json

def get_presenters(year, ANS):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    # Your code here
    data = json.load(open(f"eric/data/gg{year}_sorted_annotated.json", "r"))
    ans = dict()
    data = [i for i in data if i['award'] != "None"]
    for i in data:
        ans[i['award']] = dict()
    ans['cecil b. demille award'] = dict()
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    pipe.add_processor(ReMatch(exps=[r"(present|presented|presents|presenting)"]))

    data_w = pipe.process(deepcopy(data))
    print(len(data_w))
    for i in data_w:
        if i['award'] not in ans:
            ans[i['award']] = dict()
        for j in i['name']:
            if j[1] == ANS['winner'][i['award']] and i['award_type'] == 'name':
                continue
            ans[i['award']][j[1]] = ans[i['award']][j[1]] + 1 if j[1] in ans[i['award']] else 1
    result = dict()
    for i in ans:
        ans[i] = sorted(ans[i].items(), key=lambda x: x[1], reverse=True)
        result[i] = [j[0].lower() for j in ans[i][:2]]
        print(i, ans[i][:2])
    return result

def pre_ceremony(year):
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    AhoCorasickAutomaton("eric/data/actors.tsv", year=year)
    AhoCorasickAutomaton("eric/data/movie.tsv", year=year)
    init(year)
    # print("Pre-ceremony processing complete.")
    return

def init(year):
    coded_categories = list(json.load(open(f"data/gg{year}answers.json","r"))["award_data"].keys())
    HASH = dict()
    for category in tqdm(coded_categories):
        a = Award_Category(category)
        HASH[category] = a.entity_type if a.entity_type == 'movie' else 'name'
    Award_Category.sortRegexDict()
    print(Award_Category.award_regex_dict)
    data = json.load(open(f"data/gg{year}.json", "r"))
    sorter = TimeSort(5)
    data = sorter.process(data)
    # json.dump(data, open(f"data/gg{year}_sorted.json", "w"), indent=4)
    # data = json.load(open("data/gg2013_sorted.json", "r"))
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    tmp = AhoCorasickAutomaton("eric/data/actors.pkl", remove=False, year=year)
    tmp.automaton_type = "actor"
    pipe.add_processor(tmp)
    tmp = AhoCorasickAutomaton("eric/data/movie.pkl", remove=False, name='movie', year=year)
    tmp.automaton_type = "movie"
    pipe.add_processor(tmp)
    pipe.add_processor(NLTK(proc_num=12, remove=False))
    pipe.add_processor(Summarize(remove=False, name="name"))
    data = pipe.process(data)
    textPreprocess = PreprocessText()
    textPreprocess.process(data)
    firstPass = FirstPass(5)
    data = firstPass.extract(data)
    for i in range(len(data)):
        if not isinstance(data[i]['award'], str):
            data[i]['award'] = data[i]['award'].name
    for _, i in Award_Category.award_regex_dict.items():
        for s, e in zip(i.startIndex, i.endIndex):
            for j in range(s, e):
                if data[j]['award'] == "None":
                    data[j]['award'] = i.name
    for i in range(len(data)):
        if data[i]['award'] == "None":
            continue
        data[i]['award_type'] = HASH[data[i]['award']]
    for i in data:
        tmp = []
        for j in i['movie']: # Foreign
            flag = True
            for k in i['movie']:
                if j[1] == k[1]:
                    continue
                if j[1].lower() in k[1].lower():
                    flag = False
                    break
            if flag:
                tmp.append(j[1])
        tmp = set(tmp)
        if "Foreign" in tmp:
            tmp.remove("Foreign")
        if "Golden" in tmp:
            tmp.remove("Golden")
        if "The Red Carpet" in tmp:
            tmp.remove("The Red Carpet")
        if "The Most" in tmp:
            tmp.remove("The Most")
        if 'I Wish' in tmp:
            tmp.remove('I Wish')
        i['movie'] = list(tmp)
        i['movie'] = [[0, k] for k in i['movie'] if len(k)>3]
    json.dump(data, open(f"eric/data/gg{year}_sorted_annotated.json", "w"), indent=4)
    
    return

