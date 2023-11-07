import numpy as np
import pandas as pd
from scipy.sparse import vstack, save_npz

import sys
sys.path.append("C:\College\Projects\Multimodal Sentiment Analysis using Text and Images")
from Utils.nlp import BERT_Embeddings, tfidf_preprocessing

def GenerateEmbeddings():
    data = pd.read_csv("Data/Text/Engineered.csv")

    tweet_tfidf_train, tweet_tfidf_test, y_train, y_test = tfidf_preprocessing(data['Caption'], data['LABEL'], stratify=False, shuffle=False)

    tweet_tfidf = vstack([tweet_tfidf_train, tweet_tfidf_test])
    y = np.hstack([y_train, y_test])

    hashtags_tfidf_train, hashtags_tfidf_test, y_train, y_test = tfidf_preprocessing(data['Hashtags'].fillna(""), data['LABEL'], stratify=False, shuffle=False)

    hashtags_tfidf = vstack([hashtags_tfidf_train, hashtags_tfidf_test])

    caption_length = np.array(data['Caption Length'])
    total_hashtags = np.array(data['Total Hashtags'])

    np.save("Data/Text/TF-IDF/labels.npy", y)
    np.save("Data/Text/TF-IDF/caplength.npy", caption_length)
    np.save("Data/Text/TF-IDF/totalhash.npy", total_hashtags)

    save_npz("Data/Text/TF-IDF/captions.npz", tweet_tfidf)
    save_npz("Data/Text/TF-IDF/hashtags.npz", hashtags_tfidf)

    data['Hashtags'].fillna("", inplace=True)

    embeddings = BERT_Embeddings(data, 'LABEL', 0.2, 'bert-base-uncased', stratify=False, shuffle=False)
    embeddings.encode_data("Caption")
    embeddings.create_hidden_states()
    embeddings.create_train_test_set()

    caption_bert_train, caption_bert_test, y_train, y_test = embeddings.get_train_test()
    captions_bert = np.vstack([caption_bert_train, caption_bert_test])

    embeddings.encode_data("Hashtags")
    embeddings.create_hidden_states()
    embeddings.create_train_test_set()

    Hashtags_bert_train, Hashtags_bert_test, y_train, y_test = embeddings.get_train_test()
    Hashtags_bert = np.vstack([Hashtags_bert_train, Hashtags_bert_test])

    np.save("Data/Text/BERT/captions.npy", captions_bert)
    np.save("Data/Text/BERT/hashtags.npy", Hashtags_bert)

if __name__ == '__main__':
    GenerateEmbeddings()