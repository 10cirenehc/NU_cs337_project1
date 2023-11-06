'''Version 0.4'''
from pathlib import Path
from typing import List, Dict, Any
import re

from tqdm import tqdm
from multiprocessing import  Process
from award_filter import Award
from preprocess import PreprocessPipe, AhoCorasickAutomaton, NLTK, WordsMatch, Summarize, Duplicate
import json
from eric import gg_api
from eric.award_data import get_award_data

def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''
    data = json.load(open(f"data/gg{year}.json", "r"))
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    pipe.add_processor(WordsMatch(words=['host']))
    pipe.add_processor(AhoCorasickAutomaton("data/actors.pkl", year=year))
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
    return get_award_name(year)

def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''
    # Your code here
    path = Path(f"data/eric{year}_answers.json")
    if path.exists():
        ans = json.load(open(path, "r"))
        return {i: ans[i]['nominees'] for i in ans}
    ans = get_award_data(year)
    return {i: ans[i]['nominees'] for i in ans}

def get_winner(year):
    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    path = Path(f"data/eric{year}_answers.json")
    if path.exists():
        ans = json.load(open(path, "r"))
        return {i: ans[i]['winner'] for i in ans}
    ans = get_award_data(year)
    return {i: ans[i]['winner'] for i in ans}

def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    # Your code here
    winner = get_winner(year)
    return gg_api.get_presenters(year, dict(winner=winner))

def pre_ceremony(year):
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    _ = AhoCorasickAutomaton("data/actors.tsv", year=year)
    _ = AhoCorasickAutomaton("data/movie.tsv", year=year)
    _ = gg_api.pre_ceremony(year)
    print("Pre-ceremony processing complete.")
    return
    # p = []
    # p.append(Process(target=AhoCorasickAutomaton, args=("data/actors.tsv", "init", False, year,)))
    # p.append(Process(target=AhoCorasickAutomaton, args=("data/movie.tsv", "init2", False, year,)))
    # p.append(Process(target=gg_api.pre_ceremony, args=(year,)))
    # for i in p:
    #     i.start()
    # for i in p:
    #     i.join()
    # print("Pre-ceremony processing complete.")
    # return

def main():
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''
    year = int(input("Please input year"))
    pre_ceremony(year)
    result = dict()
    host = get_hosts(year)
    result['hosts'] = host
    print("Hosts: ", ', '.join(host))
    awards = get_awards(year)
    print("Awards: ")
    for i in awards:
        print(i)
    winner = get_winner(year)
    # print("Winner: ")
    presenters = get_presenters(year)
    # print("Presenters: ")
    nominees = get_nominees(year)
    # print("Nominees: ")
    result['award_data'] = dict()
    for i in winner:
        result['award_data'][i] = dict()
        result['award_data'][i]['winner'] = winner[i]
        result['award_data'][i]['nominees'] = nominees[i]
        result['award_data'][i]['presenters'] = presenters[i]
    json.dump(result, open(f"data/gg{year}_ours.json", "w"), indent=4)
    for i, j in result['award_data'].items():
        print(i)
        print("Winner: ", j['winner'])
        print("Nominees: ", ', '.join(j['nominees']))
        print("Presenters: ", ', '.join(j['presenters']))
        print('-'*20 + '\n')




if __name__ == '__main__':
    main()
