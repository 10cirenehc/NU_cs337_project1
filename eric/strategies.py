import nltk
from pathlib import Path
from typing import Dict, Any, Optional, List
import ahocorasick
from multiprocessing import Pool
import pickle
from tqdm import tqdm
from datetime import datetime, timezone
import numpy as np
import nltk
import pandas as pd
import regex as re
from .utils import Award_Category
from .utils import extract_movie, extract_name
import spacy

nlp = spacy.load("en_core_web_sm")


# Add co-occurances
# Add deserves, didn't deserve, won't, doesn't - negations
# Add - rule to award specific
# Add "and" rule
# Cluster names to match first names
# Filter names with one letter and only two words
# over

class Strategy:
    def __init__(self, name: str, requirements: Optional[List[str]] = None):
        self.name = name
        self.winner_regex = r"(won|wins|win|winner|goes to|acceptance speech|congrats|congratulations|congratulate|shouldn't have won)"
        self.nominee_regex = r"(nominations|nominee|nominated|rob|robbing|robbed|should\'ve been|should\'ve won|deserves|should have won|deserved|should have been|over)"
        self.presenter_regex = r"(present|presented|presents|presenting)"
        self.leftwords = ["win", "wins", "won", "nominated", "present", "shouldn\'t have won", "should have won",
                          "deserved", "deserves", "over"]
        self.rightwords = ["congrats", "congratulations", "robbing", "goes to" "should\'ve been", "robbing",
                           "should have been"]
        self.bothwords = ["winner", "acceptance speech", "congratulate", "should've", "robbed", "rob", "nominations",
                          "nominee", "presented", "presents"]

    def get_keyword_splits(self, text: str):
        winner_split = re.findall(r'(.+)\b' + self.winner_regex + r'\b(.+)', text)
        nominee_split = re.findall(r'(.+)\b' + self.nominee_regex + r'\b(.+)', text)
        presenter_split = re.findall(r'(.+)\b' + self.presenter_regex + r'\b(.+)', text)
        return winner_split, nominee_split, presenter_split

    def extract(self, data: List[Dict[str, Any]]) -> List[str]:
        raise NotImplementedError


class FirstPass(Strategy):
    def __init__(self, name: Optional[str] = None):
        super().__init__("FirstPass" if name is None else name)

    def extract(self, data: List[Dict[str, Any]]) -> list[Dict[str, Any]]:
        from .utils import Award_Category
        minute_indices = {}
        # print(len(data), type(data))

        # calculate duration
        duration = datetime.fromtimestamp(data[-1]['timestamp_ms'] / 1000, tz=timezone.utc) - datetime.fromtimestamp(
            data[0]['timestamp_ms'] / 1000, tz=timezone.utc)
        minutes = round(duration.total_seconds() / 60)

        # create a list of award names
        award_regexes = list(award for award in Award_Category.award_regex_dict.keys())
        award_regexes.append("total")

        # create a dictionary to store the number of tweets for each event over time
        freqs = {award_regex: np.zeros(minutes + 1) for award_regex in award_regexes}

        # For each event, add count based on regex matches
        # TODO - implement many to one matching
        for i in tqdm(range(len(data))):
            tweet = data[i]
            curr_minute = round((datetime.fromtimestamp(tweet['timestamp_ms'] / 1000,
                                                        tz=timezone.utc) - datetime.fromtimestamp(
                data[0]['timestamp_ms'] / 1000, tz=timezone.utc)).total_seconds() / 60)

            if i > 0:
                prev = data[i - 1]
                if curr_minute != round((datetime.fromtimestamp(prev['timestamp_ms'] / 1000,
                                                                tz=timezone.utc) - datetime.fromtimestamp(
                        data[0]['timestamp_ms'] / 1000, tz=timezone.utc)).total_seconds() / 60):
                    minute_indices[curr_minute] = i

            freqs["total"][curr_minute] += 1

            found = False
            for award_regex in Award_Category.award_regex_dict.keys():
                if bool(re.search(award_regex, tweet['clean_text'], re.IGNORECASE)) or bool(
                        re.search(award_regex, " ".join(tweet['hashtags']), re.IGNORECASE)):
                    found = True
                    names = []
                    titles = []
                    if not tweet["text"].startswith("RT"):
                        freqs[award_regex][curr_minute] += 1
                    names = extract_name(tweet)
                    if Award_Category.award_regex_dict[award_regex].entity_type == "movie":
                        titles = extract_movie(tweet, Award_Category.award_regex_dict[award_regex])
                    # Add names to names affiliated with award
                    for name in names:
                        if name in Award_Category.award_regex_dict[award_regex].affil_names:
                            Award_Category.award_regex_dict[award_regex].affil_names[name] += 1
                        else:
                            Award_Category.award_regex_dict[award_regex].affil_names[name] = 1

                    # Add titles to titles affiliated with award
                    for title in titles:
                        if title in Award_Category.award_regex_dict[award_regex].affil_titles:
                            Award_Category.award_regex_dict[award_regex].affil_titles[title] += 1
                        else:
                            Award_Category.award_regex_dict[award_regex].affil_titles[title] = 1
                    # print(tweet['text'])
                    # print(curr_minute)

                    for hashtag in tweet["hashtags"]:
                        if hashtag in Award_Category.award_regex_dict[award_regex].hashtags:
                            Award_Category.award_regex_dict[award_regex].hashtags[hashtag] += 1
                        else:
                            Award_Category.award_regex_dict[award_regex].hashtags[hashtag] = 1

                    # Add index of tweet to award.tweet_indices
                    Award_Category.award_regex_dict[award_regex].tweet_indices.append(i)
                    # Add a field to the data that stores the award name
                    data[i]["award"] = Award_Category.award_regex_dict[award_regex]
                    break

            if found == False:
                data[i]["award"] = "None"

        # Find the spike in the number of tweets for each event
        # print(freqs)
        stds = [np.std(value) for value in freqs.values()]

        window_size = round(minutes / len(Award_Category.award_regex_dict.keys())) - 2
        moving_avgs = [np.convolve(value, np.ones(window_size) / window_size, mode='full')[window_size - 1:] for value
                       in freqs.values()]
        max_intervals = [np.argsort(-moving_avg)[:2] for moving_avg in moving_avgs]

        for i in range(len(max_intervals) - 1):
            if abs(max_intervals[i][1] - max_intervals[i][0]) < window_size:
                Award_Category.award_regex_dict[award_regexes[i]].startIndex.append(minute_indices[max_intervals[i][0]])
                Award_Category.award_regex_dict[award_regexes[i]].endIndex.append(
                    minute_indices[max_intervals[i][0] + window_size])
            else:
                Award_Category.award_regex_dict[award_regexes[i]].startIndex.append(minute_indices[max_intervals[i][0]])
                Award_Category.award_regex_dict[award_regexes[i]].endIndex.append(
                    minute_indices[max_intervals[i][0] + window_size])
                Award_Category.award_regex_dict[award_regexes[i]].startIndex.append(minute_indices[max_intervals[i][1]])
                Award_Category.award_regex_dict[award_regexes[i]].endIndex.append(
                    minute_indices[max_intervals[i][1] + window_size])
        print("done with first pass")
        # find the beginning of the ceremony
        # TODO - implement a more robust way to find the beginning of the ceremony

        return data


# Full regex matching and syntactic tree to find awards and all other entities
class SemanticParseWithoutAward(Strategy):
    def __init__(self, timebox_award: Award_Category):
        super().__init__("SemanticParseWithoutAward")
        self.timebox_award = timebox_award
        self.presenters = {}
        self.nominees = {}
        self.winners = {}

    def extract(self, tweet: Dict):
        text = tweet['clean_text'].lower()
        winner_split, nominee_split, presenter_split = self.get_keyword_splits(text)
        splits = [presenter_split, nominee_split, winner_split]
        nominee_found = False
        for i in range(len(splits)):
            if nominee_found:
                break

            if splits[i] == []:
                continue

            if i == 0:
                names = extract_name(tweet)
                if len(names) == 1:
                    self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[0] in self.presenters else 1
                else:
                    index = text.find(splits[i][0][1])
                    split_word = splits[i][0][1]
                    for name in names:
                        if split_word in self.leftwords and name in text[:index]:
                            self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[
                                                                                             0] in self.presenters else 1
                        elif split_word in self.rightwords and name in text[index:]:
                            self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[
                                                                                             0] in self.presenters else 1
                        elif split_word in self.bothwords:
                            self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[
                                                                                             0] in self.presenters else 1
                continue

            if self.timebox_award.entity_type == "person":
                entities = extract_name(tweet)

            elif self.timebox_award.entity_type == "movie":
                entities = extract_movie(tweet, self.timebox_award)
                if "Rob" in entities:
                    print(tweet)

            if len(entities) == 1:
                if i == 1:
                    self.nominees[entities[0]] = self.nominees[entities[0]] + 1 if entities[0] in self.nominees else 1
                    nominee_found = True
                elif i == 2:
                    self.winners[entities[0]] = self.winners[entities[0]] + 1 if entities[0] in self.winners else 1
            else:
                index = text.find(splits[i][0][1])
                split_word = splits[i][0][1]
                for entity in entities:
                    if split_word in self.leftwords and entity in text[:index]:
                        if i == 1:
                            self.nominees[entity] = self.nominees[entity] + 1 if entity in self.nominees else 1
                            nominee_found = True

                        elif i == 2:
                            self.winners[entity] = self.winners[entity] + 1 if entity in self.winners else 1
                    elif split_word in self.rightwords and entity in text[index:]:
                        if i == 1:
                            self.nominees[entity] = self.nominees[entity] + 1 if entity in self.nominees else 1
                            nominee_found = True
                        elif i == 2:
                            self.winners[entity] = self.winners[entity] + 1 if entity in self.winners else 1
                    elif split_word in self.bothwords:
                        if i == 1:
                            self.nominees[entity] = self.nominees[entity] + 1 if entity in self.nominees else 1
                            nominee_found = True
                        elif i == 2:
                            self.winners[entity] = self.winners[entity] + 1 if entity in self.winners else 1


# Full regex matching and syntactic tree to find awards and all other entities with passed award object
class SemanticParseWithAward(Strategy):
    def __init__(self, award: Award_Category):
        super().__init__("SemanticParseWithAward")
        self.award = award
        self.presenters = {}
        self.nominees = {}
        self.winners = {}

    def extract(self, tweet: Dict):
        # add - rule
        text = tweet['clean_text'].lower()
        winner_split, nominee_split, presenter_split = self.get_keyword_splits(text)
        splits = [presenter_split, nominee_split, winner_split]
        nominee_found = False
        for i in range(len(splits)):

            if nominee_found:
                break
            if splits[i] == []:
                continue

            if i == 0:
                names = extract_name(tweet)
                if len(names) == 1:
                    self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[0] in self.presenters else 1
                else:
                    index = text.find(splits[i][0][1])
                    split_word = splits[i][0][1]
                    for name in names:
                        if split_word in self.leftwords and name in text[:index]:
                            self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[
                                                                                             0] in self.presenters else 1
                        elif split_word in self.rightwords and name in text[index:]:
                            self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[
                                                                                             0] in self.presenters else 1
                        elif split_word in self.bothwords:
                            self.presenters[names[0]] = self.presenters[names[0]] + 1 if names[
                                                                                             0] in self.presenters else 1
                continue

            if self.award.entity_type == "person":
                entities = extract_name(tweet)

            elif self.award.entity_type == "movie":
                entities = extract_movie(tweet, self.award)
            if len(entities) == 1:
                if i == 1:
                    self.nominees[entities[0]] = self.nominees[entities[0]] + 1 if entities[0] in self.nominees else 1
                    nominee_found = True
                elif i == 2:
                    self.winners[entities[0]] = self.winners[entities[0]] + 1 if entities[0] in self.winners else 1
            else:
                index = text.find(splits[i][0][1])
                split_word = splits[i][0][1]
                for entity in entities:
                    if entity in self.award.name:
                        continue
                    elif split_word in self.leftwords and entity in text[:index]:
                        if i == 1:
                            self.nominees[entity] = self.nominees[entity] + 1 if entity in self.nominees else 1
                            nominee_found = True

                        elif i == 2:
                            self.winners[entity] = self.winners[entity] + 1 if entity in self.winners else 1
                    elif split_word in self.rightwords and entity in text[index:]:
                        if i == 1:
                            self.nominees[entity] = self.nominees[entity] + 1 if entity in self.nominees else 1
                            nominee_found = True
                        elif i == 2:
                            self.winners[entity] = self.winners[entity] + 1 if entity in self.winners else 1
                    elif split_word in self.bothwords:
                        if i == 1:
                            self.nominees[entity] = self.nominees[entity] + 1 if entity in self.nominees else 1
                            nominee_found = True
                        elif i == 2:
                            self.winners[entity] = self.winners[entity] + 1 if entity in self.winners else 1


class TimeBoxCoOccurance(Strategy):
    def __init__(self, award: Award_Category):
        super().__init__("TimeBoxCoOccurance")
        self.award = award

    def extract(self, data: List[Dict[str, Any]]):
        startIndex = self.award.startIndex
        endIndex = self.award.endIndex
        for j in tqdm(range(startIndex[0], endIndex[0])):
            tweet = data[j]
            names = extract_name(tweet)
            for i in range(len(names)):
                for k in range(i + 1, len(names)):
                    if names[i] in self.award.name_co_occurance:
                        if names[k] in self.award.name_co_occurance[names[i]]:
                            self.award.name_co_occurance[names[i]][names[k]] += 1
                        else:
                            self.award.name_co_occurance[names[i]][names[k]] = 1
                    else:
                        self.award.name_co_occurance[names[i]] = {}
                        self.award.name_co_occurance[names[i]][names[k]] = 1
                    if names[k] in self.award.name_co_occurance:
                        if names[i] in self.award.name_co_occurance[names[k]]:
                            self.award.name_co_occurance[names[k]][names[i]] += 1
                        else:
                            self.award.name_co_occurance[names[k]][names[i]] = 1
                    else:
                        self.award.name_co_occurance[names[k]] = {}
                        self.award.name_co_occurance[names[k]][names[i]] = 1


class TimeBoxFreq(Strategy):
    def __init__(self, award: Award_Category):
        super().__init__("TimeBoxFreq")
        self.award = award
        award.affil_names_broad = award.affil_names.copy()
        award.affil_titles_broad = award.affil_titles.copy()
        award.hashtags_broad = award.hashtags.copy()

    def extract(self, data: List[Dict[str, Any]]):
        startIndex = self.award.startIndex
        endIndex = self.award.endIndex

        for j in tqdm(range(startIndex[0], endIndex[0])):
            tweet = data[j]
            names = extract_name(tweet)
            if self.award.entity_type == "movie":
                titles = extract_movie(tweet, self.award)
            # Add names to names affiliated with award
            for name in names:
                if name in self.award.affil_names_broad:
                    self.award.affil_names_broad[name] += 1
                else:
                    self.award.affil_names_broad[name] = 1

            # Add titles to titles affiliated with award
            if self.award.entity_type == "movie":
                for title in titles:
                    if title in self.award.affil_titles_broad:
                        self.award.affil_titles_broad[title] += 1
                    else:
                        self.award.affil_titles_broad[title] = 1

            for hashtag in tweet["hashtags"]:
                if hashtag in self.award.hashtags_broad:
                    self.award.hashtags_broad[hashtag] += 1
                else:
                    self.award.hashtags_broad[hashtag] = 1


