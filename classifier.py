import os
import pickle
from functions import getDictKey, getEpochIdentifier, separateTags
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import HashingVectorizer

# dataset folder relative path
dirname = os.path.dirname(__file__)
dataset_folder = os.path.join(dirname, 'Datasets/')
model_folder = os.path.join(dirname, 'Models/')

# definitions
PICKLE_FILE_EXTENSION = ".P"

if not os.path.exists(model_folder):
    os.mkdir(model_folder)


class Classifier:
    """
    Wrapper class for model. Exposes two functions, train and predict.
    """
    _model: OneVsOneClassifier
    _modelTags: dict
    _labelDict: dict
    _Encoder: LabelEncoder
    _hashVect: HashingVectorizer
    modelName: str
    savePath: str

    _labelAll = ["negative", "neutral", "positive"]
    _stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                  "you'd",
                  'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
                  'hers',
                  'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
                  'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                  'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                  'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                  'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                  'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                  'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                  'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 'can', 'will', 'just', 'don',
                  "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'ma']
    _auxverbs = {"wo": "will", "sha": "shall"}

    def __init__(self, makeNewModel: bool, modelName: str = None):
        """
        Constructor for the class.\n
        :param makeNewModel: Set to true if making new model. False otherwise.
        :param modelName: If makeNewModel is True, this parameter will be ignored. Otherwise, loads a model with the name supplied.
        """
        if not makeNewModel and re.search("classifier_\\d{11}(\\.P)?", modelName) is None:
            makeNewModel = True

        if not makeNewModel:
            if not modelName.endswith(PICKLE_FILE_EXTENSION):
                modelName += PICKLE_FILE_EXTENSION
            self.modelName = modelName
            self.savePath = model_folder + self.modelName
            self._model, self._modelTags = self._loadModel(self.savePath)
        else:
            self.modelName = "classifier_" + getEpochIdentifier() + PICKLE_FILE_EXTENSION
            self.savePath = model_folder + self.modelName
            self._initUser()
        self._Encoder = LabelEncoder()
        self._Encoder.fit_transform(self._labelAll)
        self._labelDict = dict(zip(self._Encoder.classes_, self._Encoder.transform(self._Encoder.classes_)))
        self._hashVect = HashingVectorizer(decode_error='replace', n_features=2 ** 20, alternate_sign=False)

        # initialize model for each sentiment by passing blank data
        for sents in self._labelAll:
            self.train(text='', sentiment=sents)

    def train(self, text: str or list, sentiment: str, tags=None, returnProcessedText=False, fromDB=False):
        """
        Wrapper method that trains the model on only a single data. Set fromDB = True if the data is from the database.

        :param text: A string or a list. List should only be used on data already preprocessed, i.e. from database.
        :param sentiment: The sentiment for the data. Accepted inputs are only 'positive', 'negative' or 'neutral'.
        :param returnProcessedText: Set to true to return the processed text. Use to save to database.
        :param fromDB: Set to true only if the data (list) is from the database.
        :return: The processed text as a list. Only returns if returnProcessedText = True.
        """
        if type(text) is str:
            text = self._preprocessText(text)
        elif type(text) is list:
            if fromDB:  # required for already processed text stored in dbs
                text = self._preprocessFromDB(text)
            text = self._getSeriesFromList(text)

        if type(tags) is str:
            tags = separateTags(tags)

        self._saveModel(self._train(text, sentiment, tags), self.savePath)

        if returnProcessedText:
            text = list(text)
            return text

    def predict(self, rawText: str):
        """
        Wrapper method that predicts a single piece of data.

        :param rawText: The raw (unprocessed) text to predict on.
        :return: The sentiment of the predicted text and the tags.
        """
        processedText = self._preprocessText(rawText)

        return self._predict(processedText)

    @staticmethod
    def _preprocessFromDB(processedText: str) -> list:
        """
        Wrapper method for when the data is from the database. Separates the string data into a list of words.

        :param processedText: The already-processed text from the database.
        :return: Returns a list to be trained.
        """
        return _separateTags(processedText)

    @staticmethod
    def _getSeriesFromList(processedTextList: list) -> pd.Series:
        """
        Gets a pandas series from a list of processed text.

        :param processedTextList: A list containing the processed text/words.
        :return: Returns a pandas series of the text.
        """
        return pd.Series(' '.join(processedTextList))

    def _preprocessText(self, pre_text):
        """
        Performs preprocessing on the text.

        :param pre_text: A unprocessed string to perform text on. Accepts tuple, list and str.
        :return: A pandas series of processed text.
        """
        if type(pre_text) is list or type(pre_text) is tuple:
            pre_text = ' '.join(pre_text)

        # remove HTMl tags, if any
        pre_text = self._strip_html(pre_text)

        # split into multiple sentences
        text_tuple = sent_tokenize(pre_text, language='english')

        # tokenize
        text_tuple = [word_tokenize(sentence, language='english') for sentence in text_tuple]

        for sentence in text_tuple:
            for index, word in enumerate(sentence):
                sentence[index] = word.lower()
                if word == "n't":
                    sentence[index] = "not"
                    if sentence[index - 1] in self._auxverbs:
                        sentence[index - 1] = self._auxverbs[sentence[index - 1]]

        # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc.
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        # remove stop words + lemmatization
        filtered_final = []
        for index, entry in enumerate(text_tuple):
            word_Lemmatized = WordNetLemmatizer()
            for word, tag in pos_tag(entry):
                if word not in self._stopwords and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                    filtered_final.append(word_Final)

        # change all text to lower case
        filtered_final = [_entry.lower() for _entry in filtered_final]

        # concatenate into one data instead of multiple individual words
        _textSeries = pd.Series(' '.join(filtered_final))

        return _textSeries

    def _initUser(self):
        """
        Wrapper method to initialize a first time user. Creates a new model and saves the model.
        """
        self._model = self._makeNewClassifier()
        self._modelTags = dict()
        self._saveModel([self._model, self._modelTags], self.savePath)

    @staticmethod
    def _strip_html(text: str):
        """
        Strips HTMl tags from a string text.

        :param text: A raw unprocessed text.
        :return: The same text with HTMl tags stripped.
        """
        soup = BeautifulSoup(text, "html.parser")
        return soup.getText()

    @staticmethod
    def _makeNewClassifier():
        """
        Makes a new model using OneVsOne classification. Model is a Logistic Regression algorithm fitted with SGD.

        :return: A OneVsOneClassifier object.
        """
        return OneVsOneClassifier(SGDClassifier(loss='log', penalty='l2', max_iter=150, learning_rate='optimal', eta0=0.00, alpha=1e-04))

    def _makeNewModelDict(self, tags):
        """
        Creates a dict and initializes into the modelTags using the tag name as the key and the corresponding model as the value.\n
        The model is a binary class where class [0] is the tag (positive class) and class [1] is not the tag (negative class).
        For N number of tags, this dict will have N number of key-value pairs. This dict is essentially multiple OneVsRestClassifiers in one list.\n
        :param tags: List of string tags.
        """
        for eachTag in tags:
            self._modelTags[eachTag] = MultinomialNB()

    @staticmethod
    def _saveModel(save_classifier, _filename):
        """
        Saves a model locally. The models are saved in the ./Models/ folder.

        :param save_classifier: The model/classifier to save.
        :param _filename: The filename to save as.
        """
        saveFile = open(_filename, 'wb')
        pickle.dump(save_classifier, saveFile)
        saveFile.close()

    @staticmethod
    def _loadModel(_filename):
        """
        Loads a model from ./Models/ folder.

        :param _filename: The filename to load the classifier from.
        :return: The loaded model/classifier.
        """
        loadFile = open(_filename, 'rb')
        classifier, modelTags = pickle.load(loadFile)
        loadFile.close()

        return classifier, modelTags

    def _train(self, _text, _sentiment, userTags):
        """
        The inner method for training the model. The model is trained using partial_fit function.

        :param _text: A preprocessed pandas series.
        :param _sentiment: The raw string sentiment, either 'positive', 'negative' or 'neutral'.
        :return: The trained model.
        """
        encSentiment = self._labelDict.get(_sentiment)

        X_new = self._hashVect.transform(_text)

        self._model.partial_fit(X_new, [encSentiment], self._Encoder.transform(self._Encoder.classes_))

        # if not training the tag model
        if userTags is None:
            return self._model, self._modelTags

        # check if modelTags contain any existing tag classes
        # if yes, check if user added any new tags
        if len(self._modelTags.keys()) != 0:
            newTags = list(set(self._modelTags.keys()) - set(userTags))

            # adds the new tag to the dict and add a new model if there are new tags
            if len(newTags) != 0:
                for eachNewTag in newTags:
                    self._modelTags[eachNewTag] = MultinomialNB()
        else:  # if modelTag doesn't contain any tag classes, initialize it
            for eachTag in userTags:
                self._modelTags[eachTag] = MultinomialNB()

        for eachTag in self._modelTags:
            if eachTag in userTags:
                self._modelTags[eachTag].partial_fit(X_new, [0], [0, 1])
            else:
                self._modelTags[eachTag].partial_fit(X_new, [1], [0, 1])

        return self._model, self._modelTags

    def _predict(self, _text):
        """
        The inner method for getting predictions from the model.

        :param _text: A preprocessed pandas series.
        :return: The string sentiment, either 'positive', 'negative' or 'neutral'.
        """
        X_new = self._hashVect.transform(_text)

        sentiment = getDictKey(self._labelDict, self._model.predict(X_new))
        retTags = list()

        for eachTag in self._modelTags:
            if self._modelTags[eachTag].predict(X_new) == [0]:
                retTags.append(eachTag)

        return sentiment, retTags
