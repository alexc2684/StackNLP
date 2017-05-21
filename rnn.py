import numpy as np

def softmax(x):
    n = np.exp(x - np.max(x))
    return n / n.sum()

class RNN:
    def __init__(self, vocab_size, hidden_dim=100, bptt_truncate=4):
        self.word_dim = vocab_size
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1.0/self.word_dim), np.sqrt(1.0/self.word_dim), (self.hidden_dim, self.word_dim))
        self.V = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), (self.word_dim, self.hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), (self.hidden_dim, self.hidden_dim))

    def forward_prop(self, x):
        steps = len(x)
        s = np.zeros((steps + 1, self.hidden_dim))
        o = np.zeros((steps, self.word_dim))
        for i in range(steps):
            s[i] = np.tanh(self.U[:,x[i]] + self.W@s[i-1])
            o[i] = softmax(self.V@s[i])
        return (o, s)

    def predict(self, x):
        return np.argmax(self.forward_prop(x)[0], axis=1)

    def cross_entropy(self, y, z):
        loss = 0
        for i in range(len(y)):
            loss -= np.log(z[i][np.argmax(y[i])])
        return loss / len(y)

    def loss(self, Y, Z):
        loss = 0
        for i in range(len(Y)):
            loss += self.cross_entropy(Y[i], Z[i])
        return loss / len(Y)
