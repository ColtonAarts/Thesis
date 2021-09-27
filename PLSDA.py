import spacy
from nltk.corpus import wordnet
from scipy.stats import multinomial
from scipy.stats import bernoulli
import nltk
import gensim.downloader
from gensim.models import Word2Vec

nlp = spacy.load('en_core_web_lg')


def format_pos(pos):
    if pos == "a" or pos == "s":
        return "ADJ"
    elif pos == "v":
        return "VERB"
    elif pos == "n":
        return "NOUN"
    elif pos == "r":
        return "ADV"
    else:
        return ''


def format_pos_wordnet(pos):
    if pos == "ADJ":
        return "s"
    elif pos == "VERB":
        return "v"
    elif pos == "NOUN":
        return "n"
    elif pos == "ADV":
        return "r"
    else:
        return ''


def base_PLSDA(sentence_original, target_pos={"ADJ", "ADV", "NOUN"}, probability_ben=0.5, strat="ST", sim_value=0.7, egi=5):
    sentence = nlp(sentence_original)
    final_results = set()
    if strat == "ST":
        lst = list()
        syn_word_dict = dict()
        possible_combinations = 1
        for word in sentence:
            if word.pos_ in target_pos and not word.is_stop:
                lst.append(word)
                synsets = wordnet.synsets(word.text)
                possible_synonyms = set()
                for syn in synsets:
                    for l in syn.lemmas():
                        if word.pos_ == format_pos(syn.pos()):
                            possible_synonyms.add(l.name())
                if word.text in possible_synonyms:
                    possible_synonyms.remove(word.text)
                syn_word_dict[word] = possible_synonyms
                possible_combinations *= (len(possible_synonyms) + 1)
        possible_combinations -= 1
        if possible_combinations > egi:
            while len(final_results) < egi:
                replacement_syns = dict()
                ben_prob = bernoulli.rvs(probability_ben, size=len(lst))
                while len(set(ben_prob)) < 2:
                    ben_prob = bernoulli.rvs(probability_ben, size=len(lst))
                for index in range(len(ben_prob)):
                    if ben_prob[index] == 1 and len(syn_word_dict[lst[index]]) > 0:
                        word = lst[index]
                        possible_synonyms = syn_word_dict[word]
                        final_replacements = list()
                        possible_synonyms = list(possible_synonyms)
                        if strat == "ST" and len(possible_synonyms) > 0:
                            array = list()
                            for ele in possible_synonyms:
                                array.append(1 / len(possible_synonyms))
                            cat = multinomial.rvs(1, array)
                            for index in range(len(cat)):
                                if cat[index] == 1:
                                    final_replacements.append(possible_synonyms[index])
                        replacement_syns[word.text] = final_replacements
                changed_sentence = sentence_original
                for ele in replacement_syns.keys():
                    changed_sentence = changed_sentence.replace(ele, replacement_syns[ele][0].replace("_", " "))
                final_results.add(changed_sentence)
    elif strat == "SFS":
        glove_vectors = gensim.downloader.load('glove-twitter-200')
        #TODO Fix
        syn_word_dict = dict()
        possible_combinations = 1
        for word in sentence:
            if word.pos_ in target_pos and not word.is_stop:
                print(word.pos_)
                format_string = word.text + "." + format_pos_wordnet(word.pos_) + ".01"
                print(format_string)
                synsets = wordnet.synsets(word.text)
                word_synset = wordnet.synset(format_string)
                possible_synonyms = set()

                for syn in synsets:
                    # secondary_sim = word_synset.path_similarity(syn)
                    # print(syn.name() + ", " + word_synset.name() + " path = " + str(secondary_sim))
                    # if syn.pos() == word_synset.pos():
                    #     secondary_sim = word_synset.lch_similarity(syn)
                    #     print(syn.name() + ", " + word_synset.name() + " lch = " + str(secondary_sim))
                    # secondary_sim = word_synset.wup_similarity(syn)
                    # print(syn.name() + ", " + word_synset.name() + " wup = " + str(secondary_sim))
                    # print(syn.lemmas())
                    # print()

                    for l in syn.lemmas():
                        print(glove_vectors.most_similar(l.name()))
                        # syn_2 = wordnet.synsets(str(l.name()))



                        #
                        # print(syn.path_similarity(syn_2))
                        synonym = nlp(l.name())
                        similarity = word.similarity(synonym)
                        if word.pos_ == format_pos(syn.pos()) and similarity > sim_value:
                            possible_synonyms.add((l.name(), similarity))
                if word.text in possible_synonyms:
                    possible_synonyms.remove(word.text)
                syn_word_dict[word] = possible_synonyms
                possible_combinations *= (len(possible_synonyms) + 1)
            possible_combinations -= 1
            print(syn_word_dict)
            if possible_combinations > egi:
                lst = list()
                for key in syn_word_dict.keys():
                    if len(lst) == 0:
                        for syn in syn_word_dict[key]:
                            second_list = list()
                            second_list.append(str(key) + ":" + str(syn[0]))
                            second_list.append(syn[1])
                            lst.append(second_list)
                        lst.append([str(key) + ":" + str(key), 1])
                    else:
                        new_list = list()
                        for syn in syn_word_dict[key]:
                            for ele in lst:
                                new_list.append([ele[0] + "," + str(key) + ":" + str(syn[0]), ele[1] + syn[1]])
                            new_list.append([lst[-1][0] + "," + str(key) + ":" + str(key), lst[-1][1] + 1])
                        lst = new_list
                print(lst)

    return final_results


# doc = "Without Shakespear's eloquent language, the update is dreary and sluggish"
doc = "a great script brought down by lousy direction"
nlp_doc = nlp(doc)

for word in nlp_doc:
    print("(" + word.text + "," + word.pos_ + ")", end=" ")
print()

print(base_PLSDA(doc, strat="SFS"))
