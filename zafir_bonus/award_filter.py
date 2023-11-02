import re
from typing import Optional, List, Any, Dict

from preprocess import Preprocessor


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
        if ans:
            return get_with_best(ans)

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
