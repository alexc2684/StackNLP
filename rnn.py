import numpy as np

def softmax(x):
    n = np.exp(x - np.max(x))
    return n / n.sum()

class RNN:
    def __init__(self, vocab_size, hidden_dim=100, bptt_truncate=4):
        self.word_dim = vocab_size
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1.0/hidden_dim), np.sqrt(1.0/hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1.0/hidden_dim), np.sqrt(1.0/hidden_dim), (hidden_dim, hidden_dim))

    def forward_prop(self, x):
        steps = len(x)
        s = np.zeros((steps + 1, self.hidden_dim))
        o = np.zeros((steps, self.word_dim))
        for i in range(steps):
            s[t] = np.tanh(self.U[:,x[t]] + W@s[t-1])
            o[t] = softmax(V@s[t])
        return (o, s)
