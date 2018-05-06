import abc

import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm

from utilities.Utilities import Utilities
from learning import logger


class BaseClassifier(object):

    def __init__(self):
        self.x = None
        self.y = None
        self.clf = None
        self.clf_best = None

    def fit(self, x, y):
        Utilities.count_occurences(y)

        self.grid_search(x, y)
        self.clf = self.construct_classifier()
        logger.info("Training...")
        return self.clf.fit(x, y)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

    def predict(self, x):
        return self.clf.predict(x)

    def grid_search(self, x, y):
        if self.clf_best is not None:
            logger.info("Best result already exist")
            return

        logger.info("Searching classifier best parameters")
        self.clf_best = self.perform_grid_search(x, y)

    @abc.abstractmethod
    def perform_grid_search(self, x, y):
        pass

    @abc.abstractmethod
    def construct_classifier(self):
        return None


class LinerClassifier(BaseClassifier):

    def construct_classifier(self):
        logger.info("Creating calibrated...")
        return CalibratedClassifierCV(svm.LinearSVC(C=self.clf_best.best_params_['C']))

    def perform_grid_search(self, x, y):
        svc = svm.LinearSVC()
        c = [0.001, 0.01, 0.1, 1, 10]
        params = [
            {'C': c}
        ]

        clf = sklearn.model_selection.GridSearchCV(svc, param_grid=params, n_jobs=-1)
        clf.fit(x, y)
        logger.info("Best parameters found:")
        logger.info(clf.best_score_)
        logger.info(clf.best_params_)
        return clf


class CalibratedClassifier(object):
    pass


class RbfClassifier(BaseClassifier):

    def construct_classifier(self):
        logger.info("Creating calibrated...")
        return CalibratedClassifierCV(
            svm.SVC(probability=True,
                    C=self.clf_best.best_params_['C'],
                    kernel="rbf",
                    gamma=self.clf_best.best_params_['gamma']))

    def perform_grid_search(self, x, y):
        clf = svm.SVC(probability=True)
        c = [0.001, 0.01, 0.1, 1, 10]
        gamma = [0.001, 0.01, 0.1, 1]

        params = [
            {'C': c, 'gamma': gamma, 'kernel': ['rbf']}
        ]

        clf = sklearn.model_selection.GridSearchCV(clf, param_grid=params, n_jobs=-1)
        clf.fit(x, y)
        logger.info("Best parameters found:")
        logger.info(clf.best_score_)
        logger.info(clf.best_params_)
        return clf