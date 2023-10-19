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
    def __init__(self, words=None, name: Optional[str] = None):
        super().__init__("WordsMatch" if name is None else name)
        if words is None:
            words = ["win", "won", "wins", "winner", 'get', 'got', 'gets', 'getting', 'gotten']
        self.words = words

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for item in tqdm(data):
            tmp = []
            for word in self.words:
                if word in item['text'].lower():
                    tmp.append(word)
            if not tmp:
                continue
            item[self.name] = tmp
            result.append(item)
        return result


class NLTK(Preprocessor):
    def __init__(self, name: Optional[str] = None, proc_num: int = 8):
        super().__init__("NLTK" if name is None else name)
        self.proc_num = proc_num

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
            if not tmp:
                continue
            item[self.name] = tmp
            result.append(item)
        return result


class AhoCorasickAutomaton(Preprocessor):
    def __init__(self, file_path: str, name: Optional[str] = None):
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
                for item in tqdm(data['primaryName']):
                    if not isinstance(item, str):
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
        super().__init__("AhoCorasickAutomaton" if name is None else name)

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for item in tqdm(data):
            # using automaton.iter is faster
            tmp = []
            # add space to the beginning to match from the beginning of the word
            for each in self.automaton.iter(f" {item['text'].lower()}"):
                tmp.append(each)
            if not tmp:
                continue
            item[self.name] = tmp
            result.append(item)
        return result

class Summarize(Preprocessor):
    def __init__(self, name: Optional[str] = None):
        super().__init__("Summarize" if name is None else name)

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for item in tqdm(data):
            tmp = []
            for i in item['NLTK']:
                for j in item['AhoCorasickAutomaton']:
                    if i in j[1]:
                        tmp.append(j)
            if not tmp:
                continue
            tmp = list(set(tmp))
            item[self.name] = []
            # remove part of names
            for i in tmp:
                flag = False
                for j in tmp:
                    if i == j:
                        continue
                    if j[1].startswith(i[1]) or j[1].endswith(i[1]):
                        flag = True
                        break
                if not flag:
                    item[self.name].append(i)
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
