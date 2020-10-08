from tqdm import tqdm
import numpy as np

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

def remove_high_low(data, ngram_counts, minimum, maximum):
    '''
    Function that will remove ngrams between certain thresholds
    args:
        data: list of list of tokens
        ngram_perc: the percentage of the dataset that each token has
        threshold: Tuple containing lower and upper threshold
    returns:
        dataset with removed tokens
    '''

    print('Minimum count: {}'.format(minimum))
    print('Maximum count: {}'.format(maximum))
    reduced = []

    for sent in data:
        cleaned = []
        for token in sent:
            if ngram_counts[token] < minimum or ngram_counts[token] > maximum:
                continue
            else:
                cleaned.append(token)
        reduced.append(cleaned)
    
    return reduced

def build_doc_vectors(data, features):

    feature_idx = {i : num for num, i in dict(enumerate(features.keys())).items()}

    vectors = []

    for doc in data:
        vector = np.zeros(len(feature_idx))
        
        for token in doc:
            vector[feature_idx[token]] += 1

        vectors.append(vector)

    return vectors

def build_features(data, n):

    feature_counts = count_n_grams(data, n)
    reduced = remove_high_low(data, feature_counts, 5, (max(feature_counts.values())- 10))

    return reduced, feature_counts