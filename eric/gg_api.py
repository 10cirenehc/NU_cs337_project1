'''Version 0.4'''
from copy import deepcopy
from typing import List, Dict, Any
import re

from tqdm import tqdm

from .award_filter import Award
from .preprocess import PreprocessPipe, AhoCorasickAutomaton, NLTK, WordsMatch, Summarize, Duplicate, FirstPass, \
    TimeSort, PreprocessText, ReMatch
from .utils import Award_Category
import json

ANS = dict()
def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''
    # Your code here
    data = json.load(open("eric/data/gg2013_sorted_annotated.json", "r"))
    data = [i for i in data if i['award'] != "None"]
    ans = dict()
    for i in data:
        ans[i['award']] = dict()
    ans['cecil b. demille award'] = dict()
    global ANS
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    pipe.add_processor(ReMatch(exps=[r"(nominations|nominee|nominated|rob|robbing|robbed|should\'ve)"]))

    data_w = pipe.process(deepcopy(data))
    print(len(data_w))
    ans = dict()
    for i in data_w:
        if i['award'] not in ans:
            ans[i['award']] = dict()
        for j in i[i['award_type']]:
            if j[1].startswith("The") and i['award_type'] == 'name':
                continue
            if j[1] == ANS['winner'][i['award']]:
                continue
            if j[1] in ANS['presenter'][i['award']] and i['award_type'] == 'name':
                continue
            ans[i['award']][j[1]] = ans[i['award']][j[1]] + 1 if j[1] in ans[i['award']] else 1
    result = dict()
    for i in ans:
        ans[i] = sorted(ans[i].items(), key=lambda x: x[1], reverse=True)
        result[i] = [j[0].lower() for j in ans[i][:5]]
        print(i, ans[i][1: 6])
    # result['cecil b. demille award'] = []
    return result

def get_winner(year):
    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    data = json.load(open("eric/data/gg2013_sorted_annotated.json", "r"))
    data = [i for i in data if i['award'] != "None"]
    ans = dict()
    for i in data:
        ans[i['award']] = dict()
    ans['cecil b. demille award'] = dict()
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    pipe.add_processor(
        ReMatch(exps=[r"(won|wins|winner|goes to|acceptance speech|congrats|congratulations|congratulate)"]))

    data_w = pipe.process(deepcopy(data))
    print(len(data_w))
    global ANS
    for i in data_w:
        if i['award'] not in ans:
            ans[i['award']] = dict()
        for j in i[i['award_type']]:
            ans[i['award']][j[1]] = ans[i['award']][j[1]] + 1 if j[1] in ans[i['award']] else 1
    result = dict()
    for i in ans:
        ans[i] = sorted(ans[i].items(), key=lambda x: x[1], reverse=True)
        result[i] = ans[i][0][0]
        print(i, ans[i][0][0])
    ANS['winner'] = result
    return result

def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    # Your code here
    data = json.load(open("eric/data/gg2013_sorted_annotated.json", "r"))
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
    global ANS
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
    ANS['presenter'] = result
    # result['cecil b. demille award'] = []
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
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''
    data = json.load(open(f"data/gg{year}.json", "r"))
    HASH = dict()
    coded_categories = list(json.load(open(f"data/gg{year}answers.json","r"))["award_data"].keys())
    for category in tqdm(coded_categories):
        a = Award_Category(category)
        HASH[category] = a.entity_type if a.entity_type == 'movie' else 'name'
    # for i in HASH:
    #     print(i, HASH[i])
    # exit(0)
    Award_Category.sortRegexDict()
    # print(Award_Category.award_regex_dict)
    sorter = TimeSort(5)
    data = sorter.process(data)

    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    tmp = AhoCorasickAutomaton("eric/data/actors.pkl", remove=False)
    tmp.automaton_type = "actor"
    pipe.add_processor(tmp)
    tmp =AhoCorasickAutomaton("eric/data/movie.pkl", remove=False, name='movie')
    tmp.automaton_type = "movie"
    pipe.add_processor(tmp)
    pipe.add_processor(NLTK(proc_num=12, remove=False))
    pipe.add_processor(Summarize(remove=False, name="name"))
    data = pipe.process(data)
    json.dump(data, open(f"eric/data/gg{year}_sorted_annotated.json", "w"), indent=4)

    data = json.load(open(f"eric/data/gg{year}_sorted_annotated.json", "r"))
    textPreprocess = PreprocessText()
    textPreprocess.process(data)
    json.dump(data, open(f"eric/data/gg{year}_sorted_annotated.json", "w"), indent=4)
    # data = sorter.process(data)
    firstPass = FirstPass(5)
    #print(timeBox.process(data))
    data = firstPass.process(data)

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
        i['movie'] = [[0, k] for k in i['movie'] if len(k)>3 and k.count(' ') > 0]
    json.dump(data, open(f"eric/data/gg{year}_sorted_annotated.json", "w"), indent=4)
    return

