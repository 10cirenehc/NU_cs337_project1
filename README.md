# NU_cs337_project1
Project 1 of CS337

## Requirements
All the required packages are contained in requirements.txt. After installing NLTK, use nltk.download('punkt'), nltk.download('averaged_perceptron_tagger'),nltk.download('vader_lexicon') to download data.

## Instructions to run
To extract host names, award names, winners, nominees, and presenters, please place the json file containing the Tweets in the data folder. The filename must be in the format "gg{year}.json". To run the program, run gg_api.py in the root directory (another gg_api.py file exists within the folder named "eric". Do not run that one).

Run the bonus content after running the main program, as the bonus extraction depends on preprocessing done in the main program. First, run "bonus.py". Then, to see Zafir's work on bonuses, run zafir_bonus.py within zafir_bonus/bonus/. The Jupyter Notebook will also ask for the year. 

## Explanation of Approach

### Preprocessing
1. The Tweets are first sorted by their timestamp
2. Using the tsv files for names and titles downloaded from imdb.com, we add each person and title (with some filters on title type, year, etc.) to an Ahocorasick data structure automation for quick string matching. This is for quickly labeling the relevent people and movie titles in each Tweet and the position of these entities, which are crucial for our analysis. For the labelling of people, a second filter using the NLTK parts of speech tagger is applied to ensure that the names found are proper nouns. 
3. Tweets text are preprocessed by converting Hashtags to natural language, removing retweet indicators, removing usernames, and applying unidecode.
4. Duplicate Tweets are removed.

### Extraction of Award Names
From Xukun: 

In addition to the preprocessing pipeline for the AhoCorasick automation, a preprocesor matching the word "best" is used to find relevant Tweets. Then, the following steps are executed: 

Pattern Matching: The script then applies regular expressions to search for patterns in text that are typically associated with award announcements, such as "award for..." or "wins...".
Extraction and Refinement: Potential award names are extracted from the matches. The list of names is then refined by filtering out n-grams that are not likely to be award names and by removing subsets of longer n-grams to avoid duplication.
Post-processing: After extracting and refining the award names, the script performs additional post-processing to clean the text, such as removing unnecessary characters and ensuring the award names meet certain criteria (e.g., contains the word 'best').
Cosine Similarity: To further refine the list, cosine similarity is calculated between the names to identify and remove very similar entries, aiming to keep the list of award names as distinct as possible.
Final Selection: The script finalizes a list of unique award names based on the text analysis and outputs them.
The get_award_name function is the entry point that uses the year of the award ceremony as a parameter to load the relevant dataset and run through the entire process, outputting the list of award names for that year.

### Extraction of Winners, Presenters, Nominees
From Eric: 

The code for this section is mainly in the eric folder in the files utils.py, strategies.py, and award_data.py. 

1. We first took the hardcoded award names and wrote a program to produce Regex strings for matching the awards. The idea was to rank the awards based on the minimum information needed to identify the award. Then, Regex strings are built to match a broad range of colloquial ways to refer to an award based on look ahead groups. The regex strings are then ranked by a hierarchy. For example, if a text matches to Best Comedy Actress, it will not match to Best Actress.

2. We then use these matches to timebox the entire event. To do so, we compute the frequencies of each award mentioned in each minute, then convolve the frequencies with a window function to get a moving average. Then, some additional statistical methods are used to find the timebox. During the first pass of the data for timeboxing, the titles, names, and hashtags affiliated with each award are also recorded and cached. 

3. Then, there are 4 different weak strategies that were implemented, two of which are very similar. SemanticParseWithoutAward and SemanticParseWithAward both use regex patters to look for patterns related to winners, nominees, and presenters. The pattern keywords are organized by whether we then need to look leftward, look rightward, or on both sides of the keyword to find our candidate answer. If there is only one named entity from our AhoCorasick automation, then that is likely the candidate answer. Some attempts were made to further filter answers with parts of speech tags and syntactic parse trees with Spacy. The former filtered out too many good answers and the latter took too long. The difference between SemanticParseWithoutAward and SemanticParseWithAward is that the first is run on all the Tweets in the timebox of the award, which has more noise but is good for presenters and nominees. The second is run only on tweets strictly correlated to awards by regex, which is very good at finding winners.

4. The third strategy was mapping out all of the co-occurances of people within a timebox. This is usseful for finding presenters and sometimes nominees, as they are often mentioned together. However, it also introduces noise with couples or other discussed pairs of people mentioned together. There was an effort to map co-occurances of people with titles and then cross-reference it with the IMDB database, but it was not ultimately finished.

5. The fourth strategy extracts all of the relevent people discussed within a timebox, which is a good source for cross referencing candidate answers from the first two methods.

6. After all strategies are run, returning candidate answers, a weighted voting system clusters the results. The winner is first found because it needs to be later removed from the nominees of the same category. There are several things we could have done better with more time: 1. Use fuzzy string matching to detect whether the nominees contain a name that is very close to the winner, 2. Use testing feedback from the autograder to better determine weights for different strategies in a sort of boosting algorithm, rather than assigning weights in a trial and error way like we are now. 

### Bonus

For the bonus content, we identified the best dressed, worst dressed, most controversially dressed, positive sentiment, negative sentiment, most discussed, and attempted to find photos corresponding to these categories. 

From Zafir:

We have done multiple bonus nlp extraction
First we observed the top 3 loved actors, we do so by reviewing what sentiments poeple had regarding all the actors. This helps tell which actors were most loved by the audience.
To this we also have the most hated actors in the award ceremony according to the audence, the actors with most hatred sentiment were extracted. This gives a perspective about the actors and can be fueled by yheir dress/speech etc.
We also have a very interesting field called the most humiliated actor due to another person, in this case we found taylor swift had a very bad reaction when adele won the award
we also got who cracked the funniest jokes, further exploration also shows that there are links to the jokes as well
another cool thing that we observed were, which parties were the most talked about, we found that democratic party, republican as well as after party is discussed, we did more nlp analysis on it as can be seen in the bonus
lastly we also covered who gave the most controversial speech, this was again done through semantic analysis
