import spacy
from PyDictionary import PyDictionary
import nltk
from nltk.corpus import wordnet
from joblib import dump, load
from sklearn.model_selection import train_test_split
import pandas
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from keras_preprocessing.text import Tokenizer
from FilesForThesis.NeuralNetwork import NeuralNetwork
from FilesForThesis.Mapping import Mapping
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report


nlp = spacy.load('en_core_web_md')
dictionary = PyDictionary()
# nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))
SYMBOLS = re.compile('[^0-9a-z #+_]')
stemmer = WordNetLemmatizer()


def clean(tweet):

    # Use this to remove hashtags since they can become nonsense words
    # trimmed_tweet = re.sub(r'(\s)#\w+', r'\1', tweet)

    # Remove all the special characters
    trimmed_tweet = re.sub(r'\W', ' ', tweet)

    # remove all single characters
    trimmed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', trimmed_tweet)

    # Remove single characters from the start
    trimmed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', trimmed_tweet)

    # Substituting multiple spaces with single space
    trimmed_tweet = re.sub(r'\s+', ' ', trimmed_tweet, flags=re.I)

    # Removes numbers
    trimmed_tweet = ''.join([i for i in trimmed_tweet if not i.isdigit()])

    # # Removing prefixed 'b'
    # trimmed_tweet = re.sub(r'^b\s+', '', trimmed_tweet)

    # Converting to Lowercase
    trimmed_tweet = trimmed_tweet.lower()

    # Lemmatization
    trimmed_tweet = trimmed_tweet.split()
    trimmed_tweet = [stemmer.lemmatize(word) for word in trimmed_tweet]
    trimmed_tweet = ' '.join(trimmed_tweet)
    return trimmed_tweet


testing_data = open("testing_25.csv", "r", encoding="utf8").read().split("\n")

testing_text = list()
testing_labels = list()

for line in testing_data:
    line = line.split(",")
    testing_text.append(line[0])
    testing_labels.append(line[-1])


MAX_NB_WORDS = 50000
# Max number of words in each tweet.
MAX_SEQUENCE_LENGTH = 250

df_no_extra = pandas.read_csv("training_25.csv")

df_no_extra['Tweet'] = df_no_extra['Tweet'].apply(clean)

# Max number of words in each tweet.
tokenizer_no_extra = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
# tokenizer_no_extra = Mapping()
tokenizer_no_extra.fit_on_texts(df_no_extra['Tweet'].values)
# Integer replacement
X_no_extra = tokenizer_no_extra.texts_to_sequences(df_no_extra['Tweet'].values)

X_no_extra = pad_sequences(X_no_extra, maxlen=MAX_SEQUENCE_LENGTH)
# Gets categorical values for the labels
Y_no_extra = pandas.get_dummies(df_no_extra['Emotion']).values

neuralNetwork_no_extra = NeuralNetwork(X_no_extra.shape[1], 4)
neuralNetwork_no_extra.fit(X_no_extra, Y_no_extra)

testing_text_no_extra = tokenizer_no_extra.texts_to_sequences(testing_text)
testing_text_no_extra = pad_sequences(testing_text_no_extra, maxlen=MAX_SEQUENCE_LENGTH)

results_no_extra = neuralNetwork_no_extra.predict(testing_text_no_extra)
indexes = ""
final_results_no_extra = list()
for prediction in results_no_extra:
    max_percent = max(prediction)
    indexes = str(prediction.tolist().index(max_percent))
    if indexes == '0':
        indexes = "anger"
    elif indexes == "1":
        indexes = "fear"
    elif indexes == "2":
        indexes = "joy"
    else:
        indexes = "sadness"
    final_results_no_extra.append(indexes)

print(classification_report(testing_labels, final_results_no_extra))

