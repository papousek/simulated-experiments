from os import path, makedirs
from util import rmse, convert_dict
import hashlib
import json
import numpy
import random


def prediction_score(probability, target_probability):
    diff = target_probability - probability
    sign = 1 if diff > 0 else -1
    normed_diff = abs(diff) / max(0.001, abs(target_probability - 0.5 + sign * 0.5))
    return 1 - normed_diff ** 2


def recommend(items_with_predictions, target_probability):
    scored = map(
        lambda (i, p): (prediction_score(p, target_probability), random.random(), i),
        items_with_predictions.items())
    return sorted(scored, reverse=True)[0][2]


def recommend_random(items_with_predictions):
    return random.choice(items_with_predictions.keys())


class Simulator:

    def __init__(self, optimal_model, model, scenario, practice_length=None, train=False):
        if practice_length is None:
            practice_length = scenario.practice_length()
        self._optimal_model = optimal_model
        self._model = model
        if train:
            self._users = scenario.train_skills()
            self._items = scenario.train_difficulties()
            self._clusters = scenario.train_clusters()
        else:
            self._users = scenario.skills()
            self._items = scenario.difficulties()
            self._clusters = scenario.clusters()
        self._train = train
        self._practice_length = scenario.practice_length()
        self._target_probability = scenario.target_probability()
        self._practice = {}
        self._rmse = {}
        self._scenario = scenario
        self._directory = None

    def simulate(self):
        self._simulate(
            self._practice,
            self._practice_length,
            recommend_fun=lambda items: recommend(items, self._target_probability))

    def save(self, directory):
        if not path.exists(directory):
            makedirs(directory)
        self._save_stats(directory)
        self._save_practice(directory)

    def load(self, directory):
        self._directory = directory
        self._load_stats()

    def rmse(self, practice_length=None):
        if practice_length is None:
            practice_length = self._practice_length
        if practice_length in self._rmse:
            return self._rmse[practice_length]
        return self._compute_rmse(self._rmse, self.get_practice(), practice_length)

    def get_practice(self):
        self._load_practice()
        self.simulate()
        return self._practice

    def get_data(self, practice_length=None):
        if practice_length is None:
            practice_length = self._practice_length
        return self._get_data(self.get_practice(), practice_length)

    def replay(self, model):
        predicted = []
        actual = []
        model.reset()
        for u in sorted(self._users.keys()):
            for item, _, correct, _ in self.get_practice()[u]:
                predicted.append(model.predict(u, item))
                model.update(u, item, correct)
                actual.append(correct)
        return rmse(predicted, actual)

    def filename(self, directory):
        return directory + '/' + self.hash()

    def hash(self):
        return hashlib.sha1(str(self)).hexdigest()

    def _save_practice(self, directory):
        filename = self.filename(directory) + '_practice.json'
        if path.exists(filename) or len(self._practice) == 0:
            return
        to_json = {
            'practice': self._practice,
        }
        with open(filename, 'w') as f:
            json.dump(to_json, f)

    def _load_practice(self):
        if self._directory is None or len(self._practice) > 0:
            return
        filename = self.filename(self._directory) + '_practice.json'
        if not path.exists(filename):
            return
        with open(filename, 'r') as f:
            to_json = json.loads(f.read())
            self._practice = convert_dict(to_json['practice'], int, list)

    def _load_stats(self):
        if self._directory is None:
            return
        filename = self.filename(self._directory) + '_stats.json'
        if not path.exists(filename):
            return
        with open(filename, 'r') as f:
            to_json = json.loads(f.read())
            self._rmse = convert_dict(to_json['rmse'], int, float)

    def _save_stats(self, directory):
        to_json = {
            'rmse': self._rmse,
        }
        filename = self.filename(directory) + '_stats.json'
        with open(filename, 'w') as f:
            json.dump(to_json, f)

    def __str__(self):
        return 'simulator, model: %s, practice length: %s, train: %s' % (
            str(self._model), self._practice_length, self._train)

    def _get_data(self, practice, practice_length):
        return [(u, p[0], p[2]) for u, ps in practice.iteritems() for p in ps[:practice_length]]

    def _simulate(self, storage, practice_length, number_of_users=None, recommend_fun=recommend):
        if len(storage) > 0:
            return
        self._model.reset()
        for u in xrange(len(self._users)):
            item_ids = set(self._items.keys())
            storage[u] = []
            for p in xrange(practice_length):
                predictions = dict(map(lambda i: (i, self._model.predict(u, i)), item_ids))
                to_practice = recommend_fun(predictions)
                real_prediction = self._optimal_model.predict(u, to_practice)
                correct = numpy.random.uniform(0, 1) < real_prediction
                self._model.update(u, to_practice, correct)
                storage[u].append((to_practice, predictions[to_practice], correct, real_prediction))
                item_ids.remove(to_practice)
            if number_of_users is not None and u >= number_of_users - 1:
                break

    def _compute_rmse(self, storage, practice, practice_length):
        if storage.get(practice_length) is None:
            flat_practice = [x for xs in practice.values() for x in xs[:practice_length]]
            storage[practice_length] = rmse(zip(*flat_practice)[1], zip(*flat_practice)[2])
        return storage[practice_length]


