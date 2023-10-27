import regex as re
import spacy
from spacy import displacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

person_words = ["role", "director", "actor", "actress", "directing"]


award_regex_dict = {
    
    r'best\s+screenplay': "best screenplay - motion picture",
    r'best\s+director' : "best director - motion picture",
    r'(?=.*comedy\s+or\s+musical)best\s+(?:performance\s+by\s+an\s+)?actress\s+(?:in\s+)?(?:a\s+)?(?:television|tv)\s+series': "best performance by an actress in a television series - comedy or musical",
    r'foreign\s+language\s+film' : "best foreign language film",
    
    r'(?=.*drama)best\s+(?:performance\s+by\s+an\s+)?actor\s+(?:in\s+)?(?:a\s+)?(?:television|tv)\s+series' : "best performance by an actor in a television series - drama",
    
}

class Award:
    award_regex_dict = {}
    
    def __init__(self, name: str):
        self.name = name
        self.nominees = []
        self.presenters = []
        self.winner = None
        self.entity_type = None
    
def detectEntityTypeFromAwardName(name: str):
    # Other types of names other than "best ___"
    if not bool(re.search(r'\b' + re.escape("best") + r'\b'),name,re.IGNORECASE):
        pass
    else:
        for word in person_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            if bool(re.search(pattern, name, re.IGNORECASE)):
                
        pattern = r'\bbest\b\s+(\w+)'
        match = re.search(pattern, name, re.IGNORECASE)
        type_indicator = match.group(1)
        
        

def regexStringFromAwardName(name: str):
    pass



{
  "hosts": [
    "amy poehler",
    "tina fey"
  ],
  "award_data": {
    "best screenplay - motion picture": {
      
    "best director - motion picture": {
      "nominees": [
        "kathryn bigelow",
        "ang lee",
        "steven spielberg",
        "quentin tarantino"
      ],
      "presenters": [
        "halle berry"
      ],
      "winner": "ben affleck"
    },
    "best performance by an actress in a television series - comedy or musical": {
      "nominees": [
        "zooey deschanel",
        "tina fey",
        "julia louis-dreyfus",
        "amy poehler"
      ],
      "presenters": [
        "aziz ansari",
        "jason bateman"
      ],
      "winner": "lena dunham"
    },
    "best foreign language film": {
      "nominees": [
        "the intouchables",
        "kon tiki",
        "a royal affair",
        "rust and bone"
      ],
      "presenters": [
        "arnold schwarzenegger",
        "sylvester stallone"
      ],
      "winner": "amour"
    },
    "best performance by an actor in a supporting role in a motion picture": {
      "nominees": [
        "alan arkin",
        "leonardo dicaprio",
        "philip seymour hoffman",
        "tommy lee jones"
      ],
      "presenters": [
        "bradley cooper",
        "kate hudson"
      ],
      "winner": "christoph waltz"
    },
    "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television": {
      "nominees": [
        "hayden panettiere",
        "archie panjabi",
        "sarah paulson",
        "sofia vergara"
      ],
      "presenters": [
        "dennis quaid",
        "kerry washington"
      ],
      "winner": "maggie smith"
    },
    "best motion picture - comedy or musical": {
      "nominees": [
        "the best exotic marigold hotel",
        "moonrise kingdom",
        "salmon fishing in the yemen",
        "silver linings playbook"
      ],
      "presenters": [
        "dustin hoffman"
      ],
      "winner": "les miserables"
    },
    "best performance by an actress in a motion picture - comedy or musical": {
      "nominees": [
        "emily blunt",
        "judi dench",
        "maggie smith",
        "meryl streep"
      ],
      "presenters": [
        "will ferrell",
        "kristen wiig"
      ],
      "winner": "jennifer lawrence"
    },
    "best mini-series or motion picture made for television": {
      "nominees": [
        "the girl",
        "hatfields & mccoys",
        "the hour",
        "political animals"
      ],
      "presenters": [
        "don cheadle",
        "eva longoria"
      ],
      "winner": "game change"
    },
    "best original score - motion picture": {
      "nominees": [
        "argo",
        "anna karenina",
        "cloud atlas",
        "lincoln"
      ],
      "presenters": [
        "jennifer lopez",
        "jason statham"
      ],
      "winner": "life of pi"
    },
    "best performance by an actress in a television series - drama": {
      "nominees": [
        "connie britton",
        "glenn close",
        "michelle dockery",
        "julianna margulies"
      ],
      "presenters": [
        "nathan fillion",
        "lea michele"
      ],
      "winner": "claire danes"
    },
    "best performance by an actress in a motion picture - drama": {
      "nominees": [
        "marion cotillard",
        "sally field",
        "helen mirren",
        "naomi watts",
        "rachel weisz"
      ],
      "presenters": [
        "george clooney"
      ],
      "winner": "jessica chastain"
    },
    "cecil b. demille award": {
      "nominees": [],
      "presenters": [
        "robert downey, jr."
      ],
      "winner": "jodie foster"
    },
    "best performance by an actor in a motion picture - comedy or musical": {
      "nominees": [
        "jack black",
        "bradley cooper",
        "ewan mcgregor",
        "bill murray"
      ],
      "presenters": [
        "jennifer garner"
      ],
      "winner": "hugh jackman"
    },
    "best motion picture - drama": {
      "nominees": [
        "django unchained",
        "life of pi",
        "lincoln",
        "zero dark thirty"
      ],
      "presenters": [
        "julia roberts"
      ],
      "winner": "argo"
    },
    "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television": {
      "nominees": [
        "max greenfield",
        "danny huston",
        "mandy patinkin",
        "eric stonestreet"
      ],
      "presenters": [
        "kristen bell",
        "john krasinski"
      ],
      "winner": "ed harris"
    },
    "best performance by an actress in a supporting role in a motion picture": {
      "nominees": [
        "amy adams",
        "sally field",
        "helen hunt",
        "nicole kidman"
      ],
      "presenters": [
        "megan fox",
        "jonah hill"
      ],
      "winner": "anne hathaway"
    },
    "best television series - drama": {
      "nominees": [
        "boardwalk empire",
        "breaking bad",
        "downton abbey (masterpiece)",
        "the newsroom"
      ],
      "presenters": [
        "salma hayek",
        "paul rudd"
      ],
      "winner": "homeland"
    },
    "best performance by an actor in a mini-series or motion picture made for television": {
      "nominees": [
        "benedict cumberbatch",
        "woody harrelson",
        "toby jones",
        "clive owen"
      ],
      "presenters": [
        "jessica alba",
        "kiefer sutherland"
      ],
      "winner": "kevin costner"
    },
    "best performance by an actress in a mini-series or motion picture made for television": {
      "nominees": [
        "nicole kidman",
        "jessica lange",
        "sienna miller",
        "sigourney weaver"
      ],
      "presenters": [
        "don cheadle",
        "eva longoria"
      ],
      "winner": "julianne moore"
    },
    "best animated feature film": {
      "nominees": [
        "frankenweenie",
        "hotel transylvania",
        "rise of the guardians",
        "wreck-it ralph"
      ],
      "presenters": [
        "sacha baron cohen"
      ],
      "winner": "brave"
    },
    "best original song - motion picture": {
      "nominees": [
        "act of valor",
        "stand up guys",
        "the hunger games",
        "les miserables"
      ],
      "presenters": [
        "jennifer lopez",
        "jason statham"
      ],
      "winner": "skyfall"
    },
    "best performance by an actor in a motion picture - drama": {
      "nominees": [
        "richard gere",
        "john hawkes",
        "joaquin phoenix",
        "denzel washington"
      ],
      "presenters": [
        "george clooney"
      ],
      "winner": "daniel day-lewis"
    },
    "best television series - comedy or musical": {
      "nominees": [
        "the big bang theory",
        "episodes",
        "modern family",
        "smash"
      ],
      "presenters": [
        "jimmy fallon",
        "jay leno"
      ],
      "winner": "girls"
    },
    "best performance by an actor in a television series - drama": {
      "nominees": [
        "steve buscemi",
        "bryan cranston",
        "jeff daniels",
        "jon hamm"
      ],
      "presenters": [
        "salma hayek",
        "paul rudd"
      ],
      "winner": "damian lewis"
    },
    "best performance by an actor in a television series - comedy or musical": {
      "nominees": [
        "alec baldwin",
        "louis c.k.",
        "matt leblanc",
        "jim parsons"
      ],
      "presenters": [
        "lucy liu",
        "debra messing"
      ],
      "winner": "don cheadle"
    }
  }
}
def 

