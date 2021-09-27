from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import wordnet
import spacy
from nltk.corpus import stopwords


nlp = spacy.load('en_core_web_md')


class Mapping:
    def __init__(self):

        self.STOPWORDS = set(stopwords.words('english'))
        self.seen_words = dict()
        self.current_num = 1
        self.secondary_count = 1

    def find_synonym(self, word, sim_value=0.7):
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

    def fit_on_texts(self, texts):
        # real_words = set()
        sequences = []
        for text in texts:
            sequence = []
            text = text.split(" ")
            for word in text:
                if word not in self.STOPWORDS:
                    # if word not in real_words and word in self.seen_words:
                    #     real_words.add(word)
                    #     self.seen_words[word] = self.current_num
                    #     self.current_num += 1
                    # real_words.add(word)
                    if word in self.seen_words:
                        sequence.append(self.seen_words[word])
                    else:
                        synonyms = self.find_synonym(word)
                        self.seen_words[word] = self.current_num
                        for syn in synonyms:
                            # if syn not in real_words:
                            #     self.seen_words[syn] = self.current_num
                            self.seen_words[syn] = self.current_num
                        sequence.append(self.current_num)
                        self.current_num += 1
            sequences.append(sequence)
        return sequences

    def texts_to_sequences(self, texts):
        sequences = []
        self.secondary_count = self.current_num
        for text in texts:
            sequence = []
            text = text.split(" ")
            for word in text:
                if word not in self.STOPWORDS:
                    if word in self.seen_words:
                        sequence.append(self.seen_words[word])
                    # else:
                    #     sequence.append(self.secondary_count)
                    #     self.secondary_count += 1
            sequences.append(sequence)
        return sequences

    def texts_to_sequences_old(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            text = text.split(" ")
            for word in text:
                if word not in self.STOPWORDS:
                    if word in self.seen_words:
                        sequence.append(self.seen_words[word])
                    else:
                        synonyms = self.find_synonym(word)
                        self.seen_words[word] = self.current_num
                        for syn in synonyms:
                            self.seen_words[syn] = self.current_num
                        sequence.append(self.current_num)
                        self.current_num += 1
            sequences.append(sequence)
        return sequences

    # mapping = Mapping()
    # lst = mapping.text_to_sequence(["This is a sentence that is angry"])
    # X = pad_sequences(lst, maxlen=250)
    # print(X)


# mapping = Mapping()
# mapping.fit_on_texts(["This is an angry sentence", "This is a happy sentence", "This is a fearful sentence",
#                       "This is a sad sentence"])
# lst = mapping.texts_to_sequences(["This is a furious sentence", "This sentence has words that are not in the training text"])
# print(lst)

