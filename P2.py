import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns  # Para visualizar la matriz de confusión

class DecisionStump:
    def __init__(self, n_features):
        self.feature_index = np.random.randint(0, n_features)
        self.threshold = np.random.uniform()
        self.polarity = 1 if np.random.rand() > 0.5 else -1

    def predict(self, X):
        feature_values = X[:, self.feature_index]
        return np.where(
            (self.polarity == 1) & (feature_values > self.threshold) |
            (self.polarity == -1) & (feature_values < self.threshold),
            1, -1)

class AdaboostBinario:
    def __init__(self, T=5, A=20):
        self.T = T
        self.A = A
        self.classifiers = []
        self.alphas = []

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        weights = np.ones(n_samples) / n_samples
        for t in range(self.T):
            best_classifier = None
            best_alpha = None
            min_error = float('inf')

            for _ in range(self.A):
                stump = DecisionStump(n_features)
                predictions = stump.predict(X)
                error = np.sum(weights[predictions != Y])

                if error < min_error:
                    min_error = error
                    best_classifier = stump
                    best_predictions = predictions

            alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))
            weights *= np.exp(-alpha * Y * best_predictions)
            weights /= np.sum(weights)
            
            self.classifiers.append(best_classifier)
            self.alphas.append(alpha)

    def predict(self, X):
        clf_preds = np.array([alpha * clf.predict(X) for alpha, clf in zip(self.alphas, self.classifiers)])
        return np.sign(np.sum(clf_preds, axis=0))

def load_mnist():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0
    return X_train, Y_train, X_test, Y_test

def balance_classes(X, Y, target_class):
    target_indices = np.where(Y == target_class)[0]
    non_target_indices = np.where(Y != target_class)[0]
    non_target_sample_size = len(target_indices)
    sampled_non_target_indices = np.random.choice(non_target_indices, non_target_sample_size, replace=False)
    balanced_indices = np.concatenate([target_indices, sampled_non_target_indices])
    np.random.shuffle(balanced_indices)
    return X[balanced_indices], np.where(Y[balanced_indices] == target_class, 1, -1)

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_mnist()

    for digit in range(10):
        X_train_balanced, Y_train_balanced = balance_classes(X_train, Y_train, digit)
        X_test_balanced, Y_test_balanced = balance_classes(X_test, Y_test, digit)

        model = AdaboostBinario(T=10, A=20)
        model.fit(X_train_balanced, Y_train_balanced)
        Y_pred_test = model.predict(X_test_balanced)

        cm = confusion_matrix(Y_test_balanced, Y_pred_test)
        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for digit {digit}')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, labels=['Non-' + str(digit), str(digit)])
        plt.yticks(tick_marks, labels=['Non-' + str(digit), str(digit)])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()