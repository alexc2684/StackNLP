import numpy as np
import pickle
from rnn import RNN

VOCAB_SIZE = 10000
UNKNOWN = "UNKNOWN_TOKEN"
START = "SENTENCE_START"
END = "SENTENCE_END"

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_train.reshape((X_train.shape[0], 1))
y_train.reshape((y_train.shape[0], 1))

with open("wti.pickle", "rb") as f:
    wti = pickle.load(f)

with open("itw.pickle", "rb") as f:
    itw = pickle.load(f)

def train(model, X_train, y_train, e=.005, epochs=100):
    losses = []
    for epoch in range(epochs):
        print(epoch/epochs, "%")
        for i in range(len(y_train)):
            model.sgd(X_train[i], y_train[i], e)

def generate_sentence(model):
    new = []
    new.append(wti[START])
    while not new[-1] == wti[END]:
        next_word_prob = model.forward_prop(new)[0]
        sampled = wti[UNKNOWN]
        while sampled == wti[UNKNOWN]:
            samples = np.random.multinomial(1, next_word_prob[-1])
            sampled = np.argmax(samples)
        new.append(sampled)
    return [itw[x] for x in new[1:-1]]

rnn = RNN(10000)
train(rnn, X_train[:1000], y_train[:1000])

with open("trained_rnn.pickle", "wb") as f:
    pickle.dump(rnn, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Finished Training")
for i in range(10):
    sent = []
    while len(sent) < 5:
        sent = generate_sentence(rnn)
    print(" ".join(sent))
