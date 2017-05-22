import numpy as np
import nltk
import theano
import csv
import itertools
import pickle

VOCAB_SIZE = 10000
UNKNOWN = "UNKNOWN_TOKEN"
START = "SENTENCE_START"
END = "SENTENCE_END"

data = []
with open("stack_overflow_comments.csv", "rt") as f:
    reader = csv.reader(f)
    data = itertools.chain(*[nltk.sent_tokenize(x[1].lower()) for x in reader])
    sentences = ["%s %s %s" % (START, x, END) for x in data]

sentences = sentences[2:]
tokenized = [nltk.word_tokenize(sentence) for sentence in sentences]
frequencies = nltk.FreqDist(itertools.chain(*tokenized))

vocabulary = frequencies.most_common(VOCAB_SIZE-1)
index_to_word = [word[0] for word in vocabulary]
index_to_word.append(UNKNOWN)

word_to_index = dict([(word, index) for index, word in enumerate(index_to_word)])

for i, sent in enumerate(tokenized):
    tokenized[i] = [word if word in word_to_index else UNKNOWN for word in sent]

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized])

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

with open("wti.pickle", "wb") as f:
    pickle.dump(word_to_index, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("itw.pickle", "wb") as f:
    pickle.dump(index_to_word, f, protocol=pickle.HIGHEST_PROTOCOL)
