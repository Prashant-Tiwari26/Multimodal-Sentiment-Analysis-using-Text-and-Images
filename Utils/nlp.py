import re
import torch
import numpy as np
import transformers
import pandas as pd
from datasets import Dataset as dt
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def pad_sequence(sequence, max_seq_length):
    if len(sequence) > max_seq_length:
        return sequence[:max_seq_length]
    else:
        padding = max_seq_length - len(sequence)
        return sequence + ['<PAD>'] * padding

def clean_caption(caption):
    cleaned_caption = re.sub(r'[#@!&?/-],', '', caption)
    cleaned_caption = cleaned_caption.lower()
    return cleaned_caption

def get_wordnet_pos(treebank_tag:str):
    """
    Map a Treebank part-of-speech tag to the corresponding WordNet part-of-speech tag.

    Parameters:
    - treebank_tag (str): The Treebank part-of-speech tag.

    Returns:
    - str: The corresponding WordNet part-of-speech tag.

    Example:
    >>> get_wordnet_pos('NN')
    'n'

    The function takes a Treebank part-of-speech tag as input and returns the corresponding WordNet
    part-of-speech tag. It can be used to convert part-of-speech tags from Treebank format (used in
    NLTK, for example) to WordNet format.

    The mapping of Treebank tags to WordNet tags is as follows:
    - 'N' (Noun) -> 'n'
    - 'J' (Adjective) -> 'a'
    - 'V' (Verb) -> 'v'
    - 'R' (Adverb) -> 'r'
    - All other cases default to 'n' (Noun).

    Note that the returned WordNet tag is a single character string.

    Please refer to the NLTK documentation for more information about part-of-speech tagging and WordNet.
    """
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class LemmTokenizer:
    """
    Tokenize and lemmatize text using NLTK's WordNetLemmatizer.

    Usage:
    tokenizer = LemmTokenizer()
    tokens = tokenizer("Example sentence")

    The LemmTokenizer class tokenizes and lemmatizes text using NLTK's WordNetLemmatizer. It provides a callable
    object, allowing it to be used as a function for tokenization and lemmatization.

    Methods:
    - __init__(): Initialize the LemmTokenizer object and create an instance of WordNetLemmatizer.
    - __call__(doc): Tokenize and lemmatize the input text.

    Example:
    >>> tokenizer = LemmTokenizer()
    >>> tokens = tokenizer("I am running in the park")
    >>> print(tokens)
    ['I', 'be', 'run', 'in', 'the', 'park']

    Note: This class requires NLTK and its dependencies to be installed.

    Please refer to the NLTK documentation for more information on tokenization, part-of-speech tagging, and lemmatization.
    """

    def __init__(self):
        """
        Initialize the LemmTokenizer object and create an instance of WordNetLemmatizer.
        """
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        """
        Tokenize and lemmatize the input text.

        Parameters:
        - doc (str): The text to be tokenized and lemmatized.

        Returns:
        - list: The list of lemmatized tokens.

        The __call__ method tokenizes the input text using word_tokenize and performs part-of-speech tagging using pos_tag.
        It then lemmatizes each token based on its part-of-speech tag using the WordNetLemmatizer and get_wordnet_pos functions.
        The resulting lemmatized tokens are returned as a list.
        """
        tokens = word_tokenize(doc)
        tokens_tags = pos_tag(tokens)
        return [self.wnl.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tokens_tags]
    

def tfidf_preprocessing(x, y, encoder=None, remove_stopwords:bool =True, test_size:float = 0.2, stratify=False, shuffle:bool=True):
    """
    Preprocesses the input data using TF-IDF vectorization.

    Args:
        x (list or array-like): Input data.
        y (list or array-like): Target labels.
        encoder (optional): Encoder for the target labels. Default is None.
        remove_stopwords (bool, optional): Flag indicating whether to remove stopwords. Default is True.
        test_size (float, optional): The proportion of the data to use for testing. Default is 0.2.
        stratify (bool, optional): Flag indicating whether to perform stratified splitting. Default is False.

    Returns:
        tuple: A tuple containing the preprocessed data and labels:
            - x_train (sparse matrix): Training data after TF-IDF vectorization.
            - x_test (sparse matrix): Testing data after TF-IDF vectorization.
            - y_train (array-like): Training labels.
            - y_test (array-like): Testing labels.

    Raises:
        TypeError: If `remove_stopwords` is not a boolean value.
        ValueError: If `test_size` is not between 0 and 1 (exclusive).
    """
    if not isinstance(remove_stopwords, bool):
        raise TypeError("Invalid input! Expected a boolean value.")
    
    if not (0 < test_size < 1):
        raise ValueError("Invalid input! The value must be between 0 and 1 (exclusive).")
    
    if encoder is not None:
        y = encoder.fit_transform(y)
    
    if remove_stopwords == True:
        stop_words = set(stopwords.words('english'))
        for i, a in enumerate(x):
            try:
                tokens = word_tokenize(a)
                filtered = [word for word in tokens if word not in stop_words]
                x[i] = ' '.join(filtered)
            except TypeError:
                continue

    if stratify==True:
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size, shuffle=shuffle, stratify=y)
    else:        
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size, shuffle=shuffle)

    vectorizer = TfidfVectorizer(tokenizer=LemmTokenizer(), sublinear_tf=True)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    
    return x_train, x_test, y_train, y_test

class BERT_Embeddings:
    """
    A class for creating BERT embeddings from a pandas DataFrame.

    Parameters:
        data (pd.core.frame.DataFrame): The input DataFrame containing the data.
        target_column (str): The name of the column in the DataFrame containing the target variable.
        test_size (float, optional): The proportion of data to use for the test set. Default is 0.2.
        model (str, optional): The BERT model to use. Default is 'distilbert-base-uncased'.
        stratify (bool, optional): Whether to stratify the train-test split based on the target_column. Default is True.

    Attributes:
        data (dt.Frame): A datatable Frame converted from the input DataFrame.
        target_column (str): The name of the target column.
        model (transformers.BertModel or transformers.DistilBertModel): The BERT model used for embedding extraction.
        tokenizer (transformers.BertTokenizer or transformers.DistilBertTokenizer): The tokenizer for the selected BERT model.
        encoded_data (dt.Frame): The tokenized and encoded data.
        hidden_states (dt.Frame): The extracted hidden states from the BERT model.
        x_train (np.ndarray): Numpy array containing the hidden states for the training set.
        y_train (np.ndarray): Numpy array containing the target values for the training set.
        x_test (np.ndarray): Numpy array containing the hidden states for the test set.
        y_test (np.ndarray): Numpy array containing the target values for the test set.

    Methods:
        extract_hidden_states(batch): Extracts hidden states from a batch using the BERT model.
        encode_data(column): Tokenizes and encodes the data for the specified column.
        create_hidden_states(batch_size): Creates hidden states for the encoded data in batches.
        create_train_test_set(): Splits the hidden states and target values into train and test sets.
        get_train_test(): Returns the training and test sets (x_train, x_test, y_train, y_test).
        get_all_data(): Returns all data, encoded data, and hidden states.

    """
    def __init__(self, data:pd.core.frame.DataFrame, target_column:str,test_size:float = 0.2, model:str = 'distilbert-base-uncased', stratify:bool=True, shuffle:bool=True) -> None:
        self.data = dt.from_pandas(data)
        self.target_column = target_column
        if "__index_level_0__" in self.data.column_names:
          self.data.remove_columns(["__index_level_0__"])

        if stratify == True:
          self.data = self.data.train_test_split(test_size=test_size, shuffle=shuffle, stratify_by_column=self.target_column)
        else:
          self.data = self.data.train_test_split(test_size=test_size, shuffle=shuffle)

        if "distil" in model:
            self.model = transformers.DistilBertModel.from_pretrained(model)
            self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(model)
        else:
            self.model = transformers.BertModel.from_pretrained(model)
            self.tokenizer = transformers.BertTokenizer.from_pretrained(model)
    
    def extract_hidden_states(self,batch):
        """
        Extracts hidden states from a batch using the BERT model.

        Parameters:
            batch: A batch of tokenized and encoded data.

        Returns:
            dict: A dictionary containing the extracted hidden states for the batch.
        """
        inputs = {k:v.to('cpu') for k, v in batch.items() if k in self.tokenizer.model_input_names}
        with torch.no_grad():
            last_hidden_state = self.model(**inputs).last_hidden_state
        return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}
    
    def encode_data(self, column):
        """
        Tokenizes and encodes the data for the specified column.

        Parameters:
            column (str): The column name for which to tokenize and encode the data.
        """
        def tokenize(batch):
          return self.tokenizer(batch[column], padding=True, truncation=True)
        self.encoded_data = self.data.map(tokenize, batched=True, batch_size=None)

    def create_hidden_states(self, batch_size:int = 500):
        """
        Creates hidden states for the encoded data in batches.

        Parameters:
            batch_size (int, optional): The batch size for processing the encoded data. Default is 500.
        """
        self.encoded_data.set_format("torch", columns=["input_ids", "attention_mask", self.target_column])
        self.hidden_states = self.encoded_data.map(self.extract_hidden_states, batched=True, batch_size=batch_size)

    def create_train_test_set(self):
        """
        Splits the hidden states and target values into train and test sets.
        """
        self.x_train = np.array(self.hidden_states['train']['hidden_state'])
        self.y_train = np.array(self.hidden_states['train'][self.target_column])
        self.x_test = np.array(self.hidden_states['test']['hidden_state'])
        self.y_test = np.array(self.hidden_states['test'][self.target_column])

    def get_train_test(self):
        """
        Returns the training and test sets.

        Returns:
            tuple: A tuple containing numpy arrays (x_train, x_test, y_train, y_test) representing the training and test data.
        """
        return self.x_train, self.x_test, self.y_train, self.y_test

    def get_all_data(self):
        """
        Returns all data, encoded data, and hidden states.

        Returns:
            tuple: A tuple containing datatable Frames representing the data, encoded data, and hidden states.
        """
        return self.data, self.encoded_data, self.hidden_states
  