from helpers import g_test

import sys
import pickle
from copy import deepcopy
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                              StackingClassifier, VotingClassifier)


ESTIMATORS = {
    'svm',
    'ann',
    'knn',
    'dtc',
    'rfc',
}

def svm(**kwargs):
    default_args = {
        'C': 10,
        'probability': True,
    }
    default_args.update(**kwargs)
    return SVC(**default_args)

def ann(**kwargs):
    default_args = {
        'solver': 'lbfgs',
        'hidden_layer_sizes': (10, 10),
        'max_iter': 10000,
    }
    default_args.update(**kwargs)
    return MLPClassifier(**default_args)

def knn(**kwargs):
    return KNeighborsClassifier(**kwargs)

def dtc(**kwargs):
    return DecisionTreeClassifier(**kwargs)

def rfc(**kwargs):
    return RandomForestClassifier(**kwargs)

class GenderClassifier:
    def __init__(self, estimators: dict):
        """`estimators` is a map of classifier name to any wanted **kwargs to be used
        while instantiating that classifier."""
        self.estimators = {}
        self.estimator_generators = {}
        try:
            for estimator_name, estimator_kwargs in estimators.items():
                est = estimator_name.split('_', 1)[0]
                assert est in ESTIMATORS, f"Unknown estimator {estimator_name}"
                estimator_generator = (
                    lambda kwargs=estimator_kwargs,
                    estimator=sys.modules[__name__].__dict__[est]: estimator(**kwargs))
                self.estimator_generators[estimator_name] = estimator_generator
        except Exception as e:
            print(e)

    def fit(self, features: dict, ys, booster=1):
        # Test each estimator against each feature.
        features_estimators = {}
        for feature_name, feature in features.items():
            features_estimators[feature_name] = {}
            for estimator_name, estimator_generator in self.estimator_generators.items():
                print(f"Testing feature '{feature_name}' with estimator '{estimator_name}': {estimator_generator()}")
                features_estimators[feature_name][estimator_name] = (
                    g_test(feature, ys, estimator_generator(), log=False))
                log = f"Accuracy: {features_estimators[feature_name][estimator_name]:.2f}%"
                print(f"\n{log}", len(log) * '-', sep='\n')
            print(len(log) * '-')
        # Assign each feature an estimator.
        best_feature_estimators_info = {}
        for feature_name, feature_estimators in features_estimators.items():
            best_estimator_name = max(feature_estimators, key=feature_estimators.get)
            best_estimator_accuracy = feature_estimators[best_estimator_name]
            best_feature_estimators_info[feature_name] = (best_estimator_name,
                                                          best_estimator_accuracy)
        print(f"Estimators selected: {best_feature_estimators_info}")
        print("Training the estimators without CV.")
        # Train each selected estimator with the whole training set of its feature.
        for feature_name, best_estimator_info in best_feature_estimators_info.items():
            estimator_name, estimator_accuracy = best_estimator_info
            estimator = self.estimator_generators[estimator_name]()
            estimator.fit(features[feature_name], ys)
            # Store a boosted accuracy of the classifier.
            self.estimators[feature_name] = (estimator, estimator_accuracy ** booster)

    def predict(self, features: dict, use_probs=False) -> float:
        assert features, "No features given!"
        # Get the prediction and contribution of each estimator using the given features.
        predictions = []
        contributions = []
        for feature_name, feature in features.items():
            if feature_name not in self.estimators.keys():
                print(f"The model wasn't trained on feature {feature_name}!")
                continue
            if use_probs:
                try:
                    prediction = self.estimators[feature_name][0].predict_proba([feature])[0][1]
                except:
                    # If the estimator has no `predict_proba`, assume 0.7 for males 0.3 for females.
                    prediction = abs(self.estimators[feature_name][0].predict([feature])[0] - 0.3)
            else:
                prediction = self.estimators[feature_name][0].predict([feature])[0]
            predictions.append(prediction)
            contributions.append(self.estimators[feature_name][1])
        # Weight every estimator's prediction based on its accuracy.
        prediction = 0
        agg = sum(contributions)
        for p, c in zip(predictions, contributions):
            prediction += p * (c / agg)
        return prediction

    def score(self, xs, ys, use_probs=False):
        males, correct, certainty, count = 0, 0, 0, len(ys)
        for features, y in zip(xs, ys):
            prediction = self.predict(features, use_probs)
            males += round(prediction)
            correct += round(prediction) == y
            certainty += abs((not round(prediction)) - prediction)
        print(f"""
              male percentage = {males * 100 / count:.2f}%
              accuracy = {correct * 100 / count:.2f}%
              avg certainty = {certainty * 100 / count:.2f}
              """)
        return correct * 100 / count

    def pickle(self, file_name='gender_classifier.pkl'):
        with open(file_name, 'wb') as clf_file:
            self_copy = deepcopy(self)
            # Get rid of the estimator generators as they can't be pickled.
            self_copy.estimator_generators = []
            pickle.dump(self_copy, clf_file)
