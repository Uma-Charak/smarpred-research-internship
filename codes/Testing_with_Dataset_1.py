import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

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

def calculate_scores(model, X_test, y_test):
    return model.score(X_test, y_test)

# Load and shuffle training data
train = load_data('reordered_I.csv')
X_train, y_train = split_data(train)

# Load test data
test_files = ['reordered_test1.csv', 'reordered_test2.csv', 'reordered_test3.csv', 'reordered_test4.csv']
test_datasets = [load_data(file) for file in test_files]
test_features = [split_data(dataset)[0] for dataset in test_datasets]
test_labels = [split_data(dataset)[1] for dataset in test_datasets]

# Define parameter grids for each model
param_grids = {
    'LinearSVC': {'model': LinearSVC(random_state=0), 'param_grid': {'C': [0.1, 1, 10, 100, 1000, 1100, 1500, 2000]}},
    'DecisionTree': {'model': DecisionTreeClassifier(random_state=0), 'param_grid': {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(2, 16)}},
    'RandomForest': {'model': RandomForestClassifier(random_state=0), 'param_grid': {'n_estimators': [100, 200, 300, 400, 700], 'max_depth': np.arange(2, 16)}},
    'MLP': {'model': MLPClassifier(random_state=0), 'param_grid': {'hidden_layer_sizes': range(2, 21), 'activation': ['relu', 'identity'], 'solver': ['lbfgs', 'sgd', 'adam'], 'learning_rate': ['constant', 'adaptive']}}
}

# Optimize parameters and calculate scores on test sets
results = {}
for model_name, params in param_grids.items():
    model = params['model']
    param_grid = params['param_grid']
    
    best_params, best_score = optimize_parameters(model, param_grid, X_train, y_train)
    best_model = model.__class__(random_state=0, **best_params)
    
    scores = []
    for features, labels in zip(test_features, test_labels):
        score = calculate_scores(best_model, features, labels)
        scores.append(score)
    
    results[model_name] = {'best_params': best_params, 'best_score': best_score, 'scores': scores}

# Print results
for model_name, result in results.items():
    print(f"{model_name} Best Parameters:")
    print(result['best_params'])
    print(f"{model_name} Best Score:")
    print(result['best_score'])
    
    for i, score in enumerate(result['scores']):
        print(f"{model_name} Test Set {i+1} Score:")
        print(score)

