import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Split sentence into array of words/tokens
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stemming = reduce word to its root form
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array
    """
    # Stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag