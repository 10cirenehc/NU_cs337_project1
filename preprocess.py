import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import ahocorasick
from multiprocessing import Pool
import pickle
from tqdm import tqdm
import nltk
import pandas as pd


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
                            if data['deathYear'][id] != '\\N' and int(data['deathYear'][id]) <= year:
                                continue
                        if len(item) < 4:
                            # ignore short names
                            continue
                        # matching from the beginning of the word
                        self.automaton.add_word(f" {str(item).lower()}", item)
                else:
                    for id, item in enumerate(tqdm(data['primaryTitle'])):
                        if not isinstance(item, str):
                            continue
                        if not (data['titleType'][id]=='movie' or data['titleType'][id].startswith('tv')):
                            continue
                        if data['startYear'][id] == '\\N':
                            continue
                        if int(data['startYear'][id]) <= year-3 or int(data['startYear'][id]) > year:
                            continue
                        if len(item) < 5:
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


class Summarize(Preprocessor):
    def __init__(self, name: Optional[str] = None, nltk_name: str = 'NLTK', acautomaton_name: str = 'AhoCorasickAutomaton',
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
