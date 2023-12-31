import json
from typing import List, Dict, Any, Tuple

from preprocess import PreprocessPipe, Duplicate, WordsMatch, AhoCorasickAutomaton, NLTK, Summarize, ReMatch


def get_best_dressed(data: List[Dict[str, Any]]) -> tuple[dict[str, list[Any]], list[dict[str, Any]]]:
    ans = dict()
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    pipe.add_processor(WordsMatch(words=["best dressed"], name="best_dressed", remove=False))
    pipe.add_processor(WordsMatch(words=["worst dressed"], name="worst_dressed", remove=False))
    pipe.add_processor(WordsMatch(words=["n't"], name="not", remove=False))
    pipe.add_processor(AhoCorasickAutomaton("data/actors.pkl", remove=True))
    pipe.add_processor(NLTK(proc_num=12, remove=False))
    pipe.add_processor(Summarize(remove=False, name="name"))
    data = pipe.process(data)
    data = [i for i in data if (i['best_dressed'] or i['worst_dressed'])]
    data = [i for i in data if not (i['best_dressed'] and i['worst_dressed'])]
    vote = dict()
    for item in data:
        for name in item['name']:
            _vote = "best_dressed" if item['best_dressed'] else "worst_dressed"
            if item['not']:
                _vote = "worst_dressed" if _vote == "best_dressed" else "best_dressed"
            if name[1] not in vote:
                vote[name[1]] = {'worst_dressed': 0, 'best_dressed': 0}
            vote[name[1]][_vote] += 1
    vote = sorted(vote.items(), key=lambda x: x[1]['best_dressed'], reverse=True)
    ans['best_dressed'] = [i[0] for i in vote[:2]]
    vote = sorted(vote, key=lambda x: x[1]['worst_dressed'], reverse=True)
    ans['worst_dressed'] = [i[0] for i in vote[:2]]
    vote = sorted(vote, key=lambda x: x[1]['best_dressed'] + x[1]['worst_dressed'], reverse=True)
    ans['most_discussed'] = [i[0] for i in vote[:2]]
    # vote = sorted(vote, key=lambda x: abs(x[1]['best_dressed'] - x[1]['worst_dressed']), reverse=True)
    # vote =
    # ans['most_controversial'] = [i[0] for i in vote[:2]]
    return ans, data

def get_user_photo(data: List[Dict[str, Any]], users: List[str]):
    ans = dict()

    # pi
    data = pipe.process(data)
    for user in users:
        ans[user] = dict()
        for item in data:
            for name in item['name']:
                if name[1] == user:
                    url = item['photo'][0]
                    if url not in ans[user]:
                        ans[user][url] = 0
                    ans[user][url] += 1
    for user in users:
        ans[user] = sorted(ans[user].items(), key=lambda x: x[1], reverse=True)
        ans[user] = [i[0] for i in ans[user][:2]]
    return ans, data


if __name__ == '__main__':
    ans, data = get_best_dressed(json.load(open("data/gg2013.json", "r")))
    data = json.load(open("data/gg2013.json", "r"))
    pipe = PreprocessPipe()
    pipe.add_processor(ReMatch(exps=[r"https://t\.co/[a-zA-Z0-9]+"], name='photo'))
    pipe.add_processor(AhoCorasickAutomaton("data/actors.pkl", remove=True))
    pipe.add_processor(NLTK(proc_num=12, remove=False))
    pipe.add_processor(Summarize(remove=False, name="name"))
    for i, j in ans.items():
        _ans, _ = get_user_photo(data, j)
        print(i, _ans)
