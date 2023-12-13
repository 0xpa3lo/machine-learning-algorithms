from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk import sent_tokenize
from nltk.corpus import stopwords

def sumamarize_text(text_string, threshold=3.9):
    # create tokens
    sentences_tekens = np.array(sent_tokenize(text_to_summarize))
    # stop words removal
    stop_words = set(stopwords.words("english"))
    # init tf_idf
    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf_matrix = tf_idf_vectorizer.fit_transform(sentences_tekens)
    # transform to matrix and sum the rows
    tf_idf_per_sentence = np.array(tf_idf_matrix.sum(axis=1)).ravel()
    # summarize tokens
    sentence_summary_indices = np.argwhere(tf_idf_per_sentence > threshold).ravel()
    return "".join(sentences_tekens[sentence_summary_indices])

