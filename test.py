import numpy as np
import pickle

UNKNOWN = "UNKNOWN_TOKEN"
START = "SENTENCE_START"
END = "SENTENCE_END"

with open("wti.pickle", "rb") as f:
    wti = pickle.load(f)

with open("itw.pickle", "rb") as f:
    itw = pickle.load(f)

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

with open("trained_rnn.pickle", "rb") as f:
    rnn = pickle.load(f)

for i in range(100):
    print("\n")
    sent = []
    while len(sent) < 5:
        sent = generate_sentence(rnn)
    print(" ".join(sent))
