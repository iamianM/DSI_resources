from nltk.corpus import stopwords
from string import printable
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize



# Text preprocessing
def run_tokenizer(content):
    '''
    Takes nx1 array where column is strings.
    Outputs list of strings
    '''
    stopwords = set(stopwords.words('english'))
    wordnet = WordNetLemmatizer()
    toks = []
    for i, c in enumerate(content):
        if len(c) != 0:
            c = ''.join([l for l in c if l in printable])
            wt = word_tokenize(c)
            c = [w for w in wt if w.lower() not in stopwords]
            lemmatized = [wordnet.lemmatize(i) for i in c]
            toks.append(' '.join(lemmatized))
    return toks

def run_TfidfVectorizer(toks):
    tfidfV = TfidfVectorizer()
    tf = tfidfV.fit_transform(toks)
    return tf, tfidfV

def run_BeautifulSoup(html_doc):
    '''
    html_doc should be from html_str.text (where html_str may be from requests.get(html))
    '''
    return BeautifulSoup(html_doc, 'html.parser')

def single_query(link, payload):
    response = requests.get(link, params=payload)
    if response.status_code != 200:
        print ('WARNING'), response.status_code
        return 0
    else:
        return response.json()
