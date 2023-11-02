from copy import deepcopy
from typing import List, Dict, Any
import re

from tqdm import tqdm

from award_filter import Award
from preprocess import PreprocessPipe, AhoCorasickAutomaton, NLTK, WordsMatch, Summarize, Duplicate, ReMatch
import json


def get_hosts(data: List[Dict[str, Any]]) -> List[str]:
    # data = json.load(open("data/gg2013.json", "r"))
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


import re


def get_award_name(data: List[Dict[str, Any]]) -> List[str]:
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    pipe.add_processor(WordsMatch(words=[':', '-', '@']))
    pipe.add_processor(WordsMatch(words=['best']))
    pipe.add_processor(AhoCorasickAutomaton("data/actors.pkl", remove=False))
    pipe.add_processor(AhoCorasickAutomaton("data/movie.pkl", remove=False, name='movie'))
    pipe.add_processor(NLTK(proc_num=12, remove=False))
    pipe.add_processor(Summarize(remove=False, name="name"))
    pipe.add_processor(Award(remove=False, name="award"))
    data = pipe.process(data)

    # json.dump(data, open("data/award.json", "w"), indent=4)


if __name__ == '__main__':
    data = json.load(open("data/gg2013_sorted_annotated.json", "r"))

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
    json.dump(data, open("data/gg2013_sorted_annotated.json", "w"), indent=4)
    exit(0)


    data = [i for i in data if i['award'] != "None"]
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    pipe.add_processor(ReMatch(exps=[r"(won|wins|winner|goes to|acceptance speech|congrats|congratulations|congratulate)"]))

    data_w = pipe.process(deepcopy(data))
    print(len(data_w))
    ans = dict()
    for i in data_w:
        if i['award'] not in ans:
            ans[i['award']] = dict()
        for j in i['name']:
            ans[i['award']][j[1]] = ans[i['award']][j[1]] + 1 if j[1] in ans[i['award']] else 1
    for i in ans:
        if not ans[i]:
            continue
        ans[i] = sorted(ans[i].items(), key=lambda x: x[1], reverse=True)
        print(i, ans[i][0][0])
    print('-'*30)

    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    pipe.add_processor(ReMatch(exps=[r"(nominations|nominee|nominated|rob|robbing|robbed|should\'ve)"]))
    data_n = pipe.process(deepcopy(data))
    ans = dict()
    for i in data_n:
        if i['award'] not in ans:
            ans[i['award']] = dict()
        for j in i['name']:
            ans[i['award']][j[1]] = ans[i['award']][j[1]] + 1 if j[1] in ans[i['award']] else 1
    for i in ans:
        if not ans[i]:
            continue
        ans[i] = sorted(ans[i].items(), key=lambda x: x[1], reverse=True)
        print(i, ans[i][:5])
    print('-'*30)
