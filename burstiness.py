import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize


def burstiness_score(text):
    sentences = sent_tokenize(text)
    lengths = [len(word_tokenize(s)) for s in sentences]
    if len(lengths) < 2:
        return 0.5
    return float(np.std(lengths) / (np.mean(lengths) + 1e-8))


# Higher burstiness implies more human-like variation.
