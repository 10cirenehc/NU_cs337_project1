from typing import List, Dict, Any
import re

from tqdm import tqdm

from award_filter import Award
from preprocess import PreprocessPipe, AhoCorasickAutomaton, NLTK, WordsMatch, Summarize, Duplicate
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
    get_award_name(json.load(open("data/gg2013.json", "r")))
    exit(0)
    data = json.load(open("data/gg2013.json", "r"))
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())  # remove duplicate sentences
    # pipe.add_processor(WordsMatch())
    # pipe.add_processor(WordsMatch(words=['best'])) # remove sentences without 'best'
    # pipe.add_processor(AhoCorasickAutomaton("data/movie.pkl", name="movie")) # remove sentences without movie name
    pipe.add_processor(AhoCorasickAutomaton("data/actors.pkl", name="name"))  # remove sentences without actor name
    pipe.add_processor(NLTK(proc_num=12))  # remove sentences without 'NNP', find actors' name
    pipe.add_processor(Summarize(acautomaton_name='name'))  # merge actors' name (NLTK and AhoCorasickAutomaton)
    data = pipe.process(data)
    for i in data:
        i['name'] = i.pop('Summarize')
    json.dump(data, open("data/gg2013_actor.json", "w"), indent=4)
    # for i in data[:30]:
    #     print(i['text'], i['AhoCorasickAutomaton'])
    # print(get_hosts(data))
    # get_award_name(data)
    exit(0)

    # for i in data:
    #     try:
    #         tmp = i['text'].split(' ')
    #         pos = 0
    #         for id, j in enumerate(tmp):
    #             if j.find('Best') != -1:
    #                 pos = id
    #                 break
    #         i['text'] = ' '.join(tmp[pos:])
    #         if i['text'].find('.') != -1:
    #             i['text'] = i['text'][:i['text'].find('.')]
    #     except:
    #         print(i['text'])
    # print("---------------------")
    # ans = dict()
    # for i in range(30):
    #     print(data[i]['text'], data[i]['Summarize'])
    # exit(0)

    # print(data[i]['text'], data[i]['Summarize'], data[i]['NLTK'], data[i]['AhoCorasickAutomaton'])
    # print(data[i]['text'])

    # for i in data:
    #     if 'Best' not in i['text']:
    #         continue
    #     tmp = i['text'].split(' ')
    #     for j in range(4, len(tmp) + 1):
    #         name = ' '.join(tmp[:j])
    #         while not name[-1].isalpha():
    #             name = name[:-1]

    # if name.lower().startswith('best director - motion picture'):
    #     flag = False
    #     print(name, i['Summarize'])
    #     for k in i['Summarize']:
    #         if k[1] in name:
    #             flag = True
    #             break
    #     print(flag)
    #     exit(0)
    #         if name.count('-') > 1:
    #             continue
    #         if name.find('!') != -1 or name.find('?') != -1 or name.find('#') != -1:
    #             continue
    #         if name.find(';') != -1 or name.find('"') != -1 or name.find('@') != -1:
    #             continue
    #         if name.find(':') != -1 or name.find('(') != -1 or name.find('(') != -1:
    #             continue
    #         if name.find("'") != -1:
    #             continue
    #         if name.find("luck") != -1:
    #             continue
    #         if name.find('&') != -1 or name.find('\\\\') != -1 or name.find('//') != -1:
    #             continue
    #         if name.find(' wins') != -1 or name.find(' at') != -1 or name.find(' is') != -1:
    #             continue
    #         if name.find(' winner') != -1 or name.find(' to') != -1 or name.find(' 2013') != -1:
    #             continue
    #         if name.find('|') != -1 or name.find('..') != -1 or name.count('-') > 1:
    #             continue
    #         if name.find(' win') != -1 or name.count('.') > 1 or name.find(' her') != -1:
    #             continue
    #         if name.find(' goes') != -1 or name.find(' to') != -1 or name.find(' 2013') != -1:
    #             continue
    #         flag = False
    #         for k in i['Summarize']:
    #             if k[1] in name:
    #                 flag = True
    #                 break
    #         if flag:
    #             continue
    #         if name not in ans:
    #             ans[name] = 1
    #         else:
    #             ans[name] += 1
    # _ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)
    # ans = []
    # print(len(_ans))
    # for i in range(50):
    #     print(_ans[i])
    # exit(0)
    # for idi, i in enumerate(tqdm(_ans)):
    #     flag = True
    #     for idj, j in enumerate(_ans):
    #         if idi == idj:
    #             continue
    #         if i[0] in j[0]:
    #             flag = False
    #             break
    #     if flag:
    #         ans.append(i)
    # x = json.load(open("data/gg2013answers.json", "r"))
    # x = [i for i in x['award_data']]
    # for i in ans:
    #     print(i)
    # print(len(ans))
    # vis = []
    # aa = 0
    # for i in x:
    #     vis.append(False)
    #     for j in ans:
    #         if i.lower() in j[0].lower():
    #             print(f"Find {i} in {j[0]}")
    #             vis[-1] = True
    #             aa += 1
    #             break
    #     # print(i)
    # print('------------', aa)
    # for i in range(len(x)):
    #     if not vis[i]:
    #         print(x[i])
