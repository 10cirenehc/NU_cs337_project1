import json

from award_filter import Award
from preprocess import PreprocessPipe, Duplicate, WordsMatch, AhoCorasickAutomaton, NLTK, Summarize, TrueAward

if __name__ == '__main__':

    data = json.load(open("data/gg2013.json", "r"))
    pipe = PreprocessPipe()
    pipe.add_processor(Duplicate())
    pipe.add_processor(WordsMatch(words=[':', '-', '@']))
    pipe.add_processor(WordsMatch(words=['best']))
    pipe.add_processor(AhoCorasickAutomaton("data/actors.pkl", remove=False))
    pipe.add_processor(AhoCorasickAutomaton("data/movie.pkl", remove=False, name='movie'))
    pipe.add_processor(NLTK(proc_num=12, remove=False))
    pipe.add_processor(Summarize(remove=False, name="name"))
    pipe.add_processor(Award(remove=False, name="award"))
    pipe.add_processor(TrueAward(name="true_award"))
    data = pipe.process(data)
    for item in data:
        print(f"{item['text']}   ->   {item['true_award']} {len(item['award'])}")


