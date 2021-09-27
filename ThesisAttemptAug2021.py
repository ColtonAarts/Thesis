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


def get_emotional_words(path):
    emotional_word_set = set()
    emotional_word_file = open(path,"r", encoding="utf8").read().split("\n")
    for line in emotional_word_file:
        line = line.split(",")
        emotional_word_set.add(line[0])
    return emotional_word_set


def find_synonym(word, sim_value=0.7):
    good_synonyms = list()
    # possible_synonyms = dictionary.synonym(word)
    synsets = wordnet.synsets(word)
    possible_synonyms = set()
    for syn in synsets:
        for l in syn.lemmas():
            possible_synonyms.add(l.name())
    word = nlp(word)
    if possible_synonyms is not None:
        for synonym in possible_synonyms:
            synonym = nlp(synonym)
            similarity = word.similarity(synonym)
            if similarity > sim_value:
                if word.text.lower() != synonym.text.lower():
                    good_synonyms.append(synonym.text)
    return good_synonyms


def synonyms_for_sentence(sentence, emotional_words):
    sentences = list()
    for word in sentence.split(" "):
        if word in emotional_words:
            print(word)
            for synonym in emotional_words[word]:
                if word.lower() != synonym.lower():
                    print(synonym)
                    sentences.append(sentence.replace(word, synonym))
    return sentences


def create_synonym_file(data_file,emotional_sysnonym_path, create_synonyms=False, emotional_word_path=None,
                        outfile = "training.csv"):
    outfile = open(outfile, "w+", encoding="utf8")
    if create_synonyms:
        words = get_emotional_words(emotional_word_path)
        emotional_synonyms = dict()
        for word in words:
            synonyms = find_synonym(word)
            if len(synonyms) > 0:
                emotional_synonyms[word] = synonyms
        dump(emotional_synonyms, emotional_sysnonym_path)
    else:
        emotional_synonyms = load(emotional_sysnonym_path)

    data_file = open(data_file, "r", encoding="utf8").read().split("\n")
    for line in data_file:
        line = line.split(",")
        outfile.write(line[0].replace("\n", " ").replace(",", " ") + ", " + line[-1] + "\n")
        sentences = synonyms_for_sentence(line[0], emotional_synonyms)
        for sentence in sentences:
            outfile.write(sentence.replace("\n", " ").replace(",", " ") + ", " + line[-1] + "\n")


def split_file(data_file_path, split_amount, outfile_training, outfile_testing, seed=42):

    df = pandas.read_csv(data_file_path)
    df['Tweet'] = df['Tweet'].apply(clean)

    X = df['Tweet'].values
    Y = df['Emotion'].values

    text_train, text_test, labels_train, labels_test = train_test_split(X, Y, test_size=split_amount, random_state=seed)

    split_training = open(outfile_training, "w+", encoding="utf8")
    split_testing = open(outfile_testing, "w+", encoding="utf8")

    for index in range(len(text_train)):
        split_training.write(text_train[index] + "," + labels_train[index] + "\n")
    for index in range(len(text_test)):
        split_testing.write(text_test[index] + "," + labels_test[index] + "\n")

emotional_word_path = "C:\\Users\\aarts\\Documents\\CPSC 371\\ThesisWork\\FilesForThesis\\lexiconFile"

# words = get_emotional_words(emotional_word_path)

#
#
# #
# # my_dict = dict()
# #
# # for word in words:
# #     synonyms = find_synonym(word)
# #     if len(synonyms) > 0:
# #         my_dict[word] = synonyms
# # dump(my_dict, "synonym_dict_7.joblib")
#
#
data_file_path = "C:\\Users\\aarts\\Documents\\CPSC 371\\ThesisWork\\FilesForThesis\\noDuplicates25Train.csv"
create_synonym_file(data_file_path, "synonym_dict_7.joblib", False, emotional_word_path, outfile="noDuplicates25TrainExpanded.csv")


# testing_data = open("testing_25.csv", "r", encoding="utf8").read().split("\n")
#
# testing_text = list()
# testing_labels = list()
#
# for line in testing_data:
#     line = line.split(",")
#     testing_text.append(line[0])
#     testing_labels.append(line[-1])
#
# df = pandas.read_csv("training.csv")
#
# df['Tweet'] = df['Tweet'].apply(clean)
#
# MAX_NB_WORDS = 50000
# # Max number of words in each tweet.
# MAX_SEQUENCE_LENGTH = 250
#
# # tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
# # tokenizer.fit_on_texts(df['Tweet'].values)
# # # Integer replacement
#
#
# tokenizer = Mapping()
# X = tokenizer.texts_to_sequences(df['Tweet'].values)
# X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
# # Gets categorical values for the labels
# Y = pandas.get_dummies(df['Emotion']).values
#
# neuralNetwork = NeuralNetwork(X.shape[1], 4)
# neuralNetwork.fit(X, Y)
#
# testing_text_extra = tokenizer.texts_to_sequences(testing_text)
# testing_text_extra = pad_sequences(testing_text_extra, maxlen=MAX_SEQUENCE_LENGTH)
# results = neuralNetwork.predict(testing_text_extra)
# print(results)
# indexes = ""
# final_results = list()
# for prediction in results:
#     max_percent = max(prediction)
#     indexes = str(prediction.tolist().index(max_percent))
#     if indexes == '0':
#         indexes = "anger"
#     elif indexes == "1":
#         indexes = "fear"
#     elif indexes == "2":
#         indexes = "joy"
#     else:
#         indexes = "sadness"
#     final_results.append(indexes)
#
# # print(classification_report(testing_labels, final_results))
#
# print(classification_report(testing_labels, final_results))

