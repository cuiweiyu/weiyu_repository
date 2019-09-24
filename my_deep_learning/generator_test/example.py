import numpy as np
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, GridSearchCV


def batch_generator(X, y, batch_size):
    n_splits = len(X) // (batch_size - 1)
    X = np.array_split(X, n_splits)
    y = np.array_split(y, n_splits)

    while True:
        for i in range(len(X)):
            X_batch = []
            y_batch = []
            for ii in range(len(X[i])):
                X_batch.append(X[i][ii].toarray().astype(np.int8))  # conversion sparse matrix -> np.array
                y_batch.append(y[i][ii])
            yield (np.array(X_batch), np.array(y_batch))


def build_model(n_hidden=32):
    model = Sequential([
        Dense(n_hidden, input_dim=4),
        Activation("relu"),
        Dense(n_hidden),
        Activation("relu"),
        Dense(3),
        Activation("sigmoid")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


iris = datasets.load_iris()
X = iris["data"]
y = iris["target"].flatten()

param_grid = {
    "n_hidden": np.array([4, 8, 16]),
    "nb_epoch": np.array(range(50, 61, 5))
}

model = KerasClassifier(build_fn=build_model, verbose=0)
skf = StratifiedKFold(n_splits=5).split(X, y)  # this yields (train_indices, test_indices)

grid = GridSearchCV(model, param_grid, cv=skf, verbose=2, n_jobs=4)
grid.fit(X, y)

print(grid.best_score_)
print(grid.cv_results_["params"][grid.best_index_])
