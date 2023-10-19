from preprocess import PreprocessPipe, AhoCorasickAutomaton, NLTK, WordsMatch, Summarize
import json

if __name__ == '__main__':
    pipe = PreprocessPipe()
    # pipe.add_processor(WordsMatch())
    pipe.add_processor(NLTK(proc_num=12))
    pipe.add_processor(AhoCorasickAutomaton("data/actors.pkl"))
    pipe.add_processor(Summarize())
    data = json.load(open("data/gg2013.json", "r"))
    data = pipe.process(data)
    for i in range(30):
        print(data[i]['text'], data[i]['Summarize'])
        # print(data[i]['text'], data[i]['Summarize'], data[i]['NLTK'], data[i]['AhoCorasickAutomaton'])
        # print(data[i]['Summarize'])
