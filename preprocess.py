from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import string
from nltk.corpus import stopwords
from nltk.util import bigrams, trigrams

lemmatizer = WordNetLemmatizer()

def tokenize(data):
    
    # Will take a list of stings

    tokenized = []
    for sent in tqdm(data):
        tokenized.append(word_tokenize(sent.lower()))

    return tokenized

def lemmatize(data):

    # will take a list of tokens 

    output = []
    for sent in tqdm(data):

        lemmatized = [] 
        for token in sent:
            lemmatized.append(lemmatizer.lemmatize(token))
        output.append(lemmatized)

    return output

def stopword_removal(data):
    punctuation = string.punctuation
    stopWords = list(stopwords.words('english'))

    cleaned = []
    
    for sent in tqdm(data):
        clean = []

        for token in sent:
            if token in punctuation or token in stopWords:
                continue
            
            else:
                clean.append(token)

        cleaned.append(clean)

    return cleaned

def count_n_grams(data, n):

    ngram_counts = {}

    for sent in data:

        if n == 1:
            grams = sent
        elif n == 2:
            grams = bigrams(sent)
        elif n == 3:
            grams = trigrams(sent)

        for n_gram in grams:
            if n_gram not in ngram_counts:
                ngram_counts[n_gram] = 1
            else:
                ngram_counts[n_gram] += 1

    return ngram_counts

def find_high_low_frequency_ngrams(ngram_counts):

    total_ngrams = sum(ngram_counts.values())

    ngram_perc = {}
    for wrd, count in ngram_counts.items():
        ngram_perc[wrd] = count/total_ngrams

    return ngram_perc

def remove_high_low(data, ngram_perc, threshold):
    '''
    Function that will remove ngrams between certain thresholds
    args:
        data: list of list of tokens
        ngram_perc: the percentage of the dataset that each token has
        threshold: Tuple containing lower and upper threshold
    returns:
        dataset with removed tokens
    '''

    pass

def preprocess(data):

    tokens = tokenize(data)
    lemmas = lemmatize(tokens)
    cleaned = stopword_removal(lemmas)
    #ngram_counts = count_n_grams(cleaned, 1)
    #high_low_range = find_high_low_frequency_ngrams(ngram_counts)
    #normalised_data = remove_high_low(cleaned, high_low_range, threshold)
    return cleaned

