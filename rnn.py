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

    def bptt(self, x, y):
        o, s = self.forward_prop(x)
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)
        do = o
        do[np.arange(len(y)), y] -= 1

        for t in np.arange(len(y))[::-1]:
            dV += np.outer(do[t], s[t].T)
            dt = self.V.T@(do[t])*(1-(s[t]**2))
            for step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dW += np.outer(dt, s[step-1])
                dU[:,x[step]] += dt
                dt = self.W.T@(dt)*(1 - s[step-1]**2)
        return (dU, dV, dW)

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

    def sgd(self, x, y, e):
        dU, dV, dW = self.bptt(x, y)
        self.U -= e*dU
        self.V -= e*dV
        self.W -= e*dW
