import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_word_counts = {}
        self.class_counts = {}
        self.vocab_size = X.shape[1]

        for c in self.classes:
            X_c = X[y == c]
            self.class_counts[c] = X_c.shape[0]
            self.class_word_counts[c] = np.sum(X_c, axis=0) + 1  # Laplace smoothing

        self.class_probs = {
            c: np.log(self.class_counts[c] / len(y)) for c in self.classes
        }
        epsilon = 1e-9
        self.word_probs = {
            c: np.log(self.class_word_counts[c] / (np.sum(self.class_word_counts[c]) + epsilon))
            for c in self.classes
        }

    def predict(self, X):
        result = []
        for x in X:
            log_probs = {}
            for c in self.classes:
                log_prob = self.class_probs[c] + x.dot(self.word_probs[c].T).item()
                log_probs[c] = log_prob
            result.append(max(log_probs, key=log_probs.get))
        return result

    def get_word_priors(self, X):
        word_priors = []
        for x in X:
            word_prob = {}
            for c in self.classes:
                log_prob = self.class_probs[c] + x.dot(self.word_probs[c].T).item()
                word_prob[c] = log_prob
            word_priors.append(word_prob)
        return word_priors