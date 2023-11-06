

import json
import re
from typing import Optional, List, Any, Dict

from preprocess import Preprocessor, PreprocessPipe, Duplicate, WordsMatch, AhoCorasickAutomaton, NLTK, Summarize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer()
    corpus = [text1, text2]
    vectors = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(vectors)
    return similarity[0][1]

def get_sorted_ngram_frequencies(data):
    cnt = dict()
    for item in data:
        tmp = ' '.join(item)
        cnt[tmp] = cnt.get(tmp, 0) + 1
    tmp = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
    return [[i[0].split(), i[1]] for i in tmp]

def filter_award_sorted(data):
    return [i for i in data if i[0][0] == 'best' and len(i[0]) > 2]
def remove_subset(data):
    ans = []
    for i in range(len(data)):
        flag = False
        for j in range(i + 1, len(data)):
            if ' '.join(data[i][0]) in ' '.join(data[j][0]):
                flag = True
                break
        if not flag:
            ans.append(data[i][0])
    return ans

def get_with_best(data: List[str]) -> List[str]:
    return [i for i in data if 'best' in i.lower()]


class Award(Preprocessor):
    def __init__(self, name: Optional[str] = None, remove: bool = True):
        super().__init__("Award" if name is None else name)
        self.remove = remove

    def check(self, data):
        text = data['text'].lower()
        ans = []
        # award for ...
        ans += re.findall(r"award for (.*?) ", text)
        ans += re.findall(r"rt @\w+: (.*?) is", text)
        ans += re.findall(r"rt @\w+: (.*?):", text)
        ans += re.findall(r"for the (.*?) award", text)
        ans += re.findall(r"presents the (.*?) to", text)
        ans += re.findall(r"presented the (.*?) to", text)
        ans += re.findall(r"present \w+ for (.*?)", text)
        ans += re.findall(r"wins .*? for (.*?)", text)
        ans += re.findall(r"wins (.*?)", text)
        ans += re.findall(r"won .*? for (.*?)", text)
        ans += re.findall(r"(.*?) is awarded to ", text)
        ans += re.findall(r"the (.*?) award goes to", text)
        if ans:
            return get_with_best(list(set(ans)))

        # actor - ...
        if data['name']:
            for i in data['name']:
                if i[0] == len(text) - 1:
                    continue
                if text[i[0] + 1:].strip().startswith('-') or text[i[0] + 1:].strip().startswith(':') or text[i[
                                                                                                                  0] + 1:].strip().startswith(
                    ','):
                    ans.append(text[i[0] + 2:].strip())
            for i in data['name']:
                start = i[0] - len(i[1])
                if start <= 0:
                    continue
                if text[:start].strip().endswith(': ') or text[:start].strip().endswith(':'):
                    ans.append(text[: start].strip())
                    continue
                if text[:start].strip().endswith('- ') or text[:start].strip().endswith('-'):
                    ans.append(text[: start].strip())
        if ans:
            return get_with_best(ans)

        if data['movie']:
            for i in data['movie']:
                if i[0] == len(text) - 1:
                    continue
                if text[i[0] + 1:].strip().startswith('-') or text[i[0] + 1:].strip().startswith(':') or text[i[
                                                                                                                  0] + 1:].strip().startswith(
                    ','):
                    ans.append(text[i[0] + 2:].strip())
            for i in data['movie']:
                start = i[0] - len(i[1])
                if start <= 0:
                    continue
                if text[:start].strip().endswith(': ') or text[:start].strip().endswith(':'):
                    ans.append(text[: start].strip())
                    continue
                if text[:start].strip().endswith('- ') or text[:start].strip().endswith('-'):
                    ans.append(text[: start].strip())
        if ans:
            return get_with_best(ans)
        return ans

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ans = []
        for item in data:
            item[self.name] = self.check(item)
            if not item[self.name] and self.remove:
                continue
            ans.append(item)
        return ans


def get_award_name(year):
    ori_data = json.load(open(f"data/gg{year}.json", "r"))
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    pipe.add_processor(WordsMatch(words=[':', '-', '@']))
    pipe.add_processor(WordsMatch(words=['best']))
    pipe.add_processor(AhoCorasickAutomaton("data/actors.pkl", remove=False, year=year))
    pipe.add_processor(AhoCorasickAutomaton("data/movie.pkl", remove=False, name='movie', year=year))
    pipe.add_processor(NLTK(proc_num=12, remove=False))
    pipe.add_processor(Summarize(remove=False, name="name"))
    pipe.add_processor(Award(remove=False, name="award"))
    data = pipe.process(ori_data)
    names = []
    for i in data:
        if not i['award']:
            continue
        names += i['award']
    names = [i.split(' ') for i in names]
    sorted_awards = get_sorted_ngram_frequencies(names)
    sorted_awards = filter_award_sorted(sorted_awards)
    sorted_awards = remove_subset(sorted_awards)
    # sorted_awards = [i for i in sorted_awards if i[1].lower() != 'actor' and i[1].lower() != 'actress']
    ans = [dict(text=' '.join(i)) for i in sorted_awards]
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    pipe.add_processor(AhoCorasickAutomaton("data/actors.pkl", remove=False, name='name', year=year))
    ans = pipe.process(ans)
    for id in range(len(ans)):
        for j in ans[id]['name']:
            if j[1].lower() in ans[id]['text']:
                ans[id]['text'] = ans[id]['text'].replace(j[1].lower(), '')
                # print(j[1])
        while ans[id]['text'].count('-') >= 2:
            ans[id]['text'] = ans[id]['text'][:ans[id]['text'].rfind('-')]
        while not ans[id]['text'][-1].isalpha():
            ans[id]['text'] = ans[id]['text'][:-1]
    ans = [i['text'] for i in ans]
    data = []
    for i in ans:
        text = i
        text = text.replace('golden globes', '')
        text = text.replace('(', '')
        text = text.replace(')', '')
        text = text.replace('/', ' or ')
        text = text.replace('  ', ' ')
        text = text.replace(' tv ', ' television ')
        while not text[-1].isalpha():
            text = text[:-1]
        flag = False
        for j in text:
            if not j.isalpha() and j != ' ':
                if j not in [',', '-']:
                    flag = True
                    break
        if flag:
            continue
        if 'actor' in text:
            if 'actor in' in text:
                data.append(text)
        elif 'actress' in text:
            if 'actress in' in text:
                data.append(text)
        else:
            data.append(text)
        for k in range(len(data) - 1):
            if calculate_cosine_similarity(data[k], data[-1]) >= 0.99:
                if 'actor' not in data[-1] and 'actress' not in data[-1]:
                    if '-' in data[-1]:
                        data.pop(k)
                    else:
                        data.pop()
                    break
        if len(data) >= 26:
            break

    result = set([])
    for i in data:
        result.add(i)
        if 'actor' in i:
            text = i.replace('actor', 'actress')
            result.add(text)
        if 'actress' in i:
            text = i.replace('actress', 'actor')
            result.add(text)
    tmp = list(result)
    for i in tmp:
        for j in tmp:
            if i == j:
                continue
            if calculate_cosine_similarity(i, j) > 0.99:
                try:
                    if '-' in i:
                        result.remove(j)
                    else:
                        result.remove(i)
                except:
                    pass
    result = list(result)
    tmp = []
    for i in result:
        flag = False
        for j in result:
            if i == j:
                continue
            if i in j:
                flag = True
                break
        if not flag:
            tmp.append(i)
    result = tmp
    result = [i for i in result if i.count(' ') >= 2 and len(i)>=17]
    for i in result:
        print(i)
    print(len(result))
    return result


if __name__ == '__main__':
    get_award_name()
