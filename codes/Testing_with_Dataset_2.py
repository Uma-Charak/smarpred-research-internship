#Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def load_data(filename):
    df = pd.read_csv(filename, sep=",", header='infer')
    df = df.drop(columns=['AT_TA_percent'])
    return shuffle(df, random_state=0)

def split_data(df):
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    return X, y

def optimize_parameters(model, param_grid, X_train, y_train):
    optimal_params = GridSearchCV(model, param_grid=param_grid, cv=4, scoring='accuracy')
    optimal_params.fit(X_train, y_train)
    return optimal_params.best_params_, optimal_params.best_score_

def calculate_scores(model, test_sets):
    scores = []
    for X_test, y_test in test_sets:
        score = model.score(X_test, y_test)
        scores.append(score)
    return scores

# Load and shuffle training data
train = load_data('reordered_II.csv')
X_train, y_train = split_data(train)

# Load test data
test_files = ['reordered_test1.csv', 'reordered_test2.csv', 'reordered_test3.csv', 'reordered_test4.csv']
test_sets = [split_data(load_data(file)) for file in test_files]

# Define models and parameter grids
models = {
    'LSVC': (LinearSVC(random_state=0), {'C': [0.1, 1, 10, 100, 1000, 1100, 1500, 2000]}),
    'DT': (DecisionTreeClassifier(random_state=0), {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(2, 16)}),
    'RF': (RandomForestClassifier(random_state=0), {'n_estimators': [100, 200, 300, 400, 700], 'max_depth': np.arange(2, 16)}),
    'MLP': (MLPClassifier(random_state=0), {'hidden_layer_sizes': range(2, 21), 'activation': ['relu', 'identity'], 'solver': ['lbfgs', 'sgd', 'adam'], 'learning_rate': ['constant', 'adaptive']})
}

# Optimize parameters and calculate scores
results = {}
for model_name, (model, param_grid) in models.items():
    best_params, best_score = optimize_parameters(model, param_grid, X_train, y_train)
    model_best = model.__class__(**best_params)
    scores = calculate_scores(model_best, test_sets)
    results[model_name] = (best_params, best_score, scores)

# Print results
for model_name, (best_params, best_score, scores) in results.items():
    print(f"{model_name} Best Parameters:")
    print(best_params)
    print(f"{model_name} Best Score:")
    print(best_score)
    for i, score in enumerate(scores):
        print(f"{model_name} Test Set {i+1} Score:")
        print(score)
    print()
