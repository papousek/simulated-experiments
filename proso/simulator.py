from os import path, makedirs
from .util import rmse, convert_dict
import hashlib
import json
import numpy
import random
from functools import reduce


def prediction_score(probability, target_probability):
    diff = target_probability - probability
    sign = 1 if diff > 0 else -1
    normed_diff = abs(diff) / max(0.001, abs(target_probability - 0.5 + sign * 0.5))
    return 1 - normed_diff ** 2


def recommend(items_with_predictions, target_probability):
    scored = [(prediction_score(i_p[1], target_probability), random.random(), i_p[0]) for i_p in list(items_with_predictions.items())]
    return sorted(scored, reverse=True)[0][2]


def recommend_random(items_with_predictions):
    return random.choice(list(items_with_predictions.keys()))


class Simulator:

    def __init__(self, optimal_model, model, scenario, practice_length=None, train=False, target_probability=None):
        if practice_length is None:
            practice_length = scenario.practice_length()
        if target_probability is None:
            target_probability = scenario.target_probability()
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
        self._target_probability = target_probability
        self._practice = {}
        self._rmse = {}
        self._jaccard = {}
        self._intersection = {}
        self._replay = {}
        self._number_of_answers = None
        self._scenario = scenario
        self._directory = None
        self._stats_loaded = False

    def simulate(self):
        self._simulate(
            self._practice,
            self._practice_length,
            recommend_fun=lambda items: recommend(items, self._target_probability))

    def number_of_answers(self):
        if self._number_of_answers is None:
            result = dict([(i_d[0], 0) for i_d in list(self._scenario.difficulties().items())])
            def _reducer(acc, i):
                acc[i] += 1
                return acc
            reduce(
                _reducer,
                [x[0] for x in [x for xs in list(self.get_practice().values()) for x in xs]], result)
            self._number_of_answers = result
        return self._number_of_answers

    def jaccard(self, practice_length=None, baseline=None):
        if practice_length is None:
            practice_length = self._practice_length
        if baseline is None:
            baseline = self._scenario.optimal_simulator()
        jaccard_key = '%s:%s' % (baseline.hash(), practice_length)
        result = self._jaccard.get(jaccard_key)
        if result is None:
            jaccard = []
            for u in range(self._scenario.number_of_users()):
                first_set = set(list(zip(*baseline.get_practice()[u]))[0][:practice_length])
                second_set = set(list(zip(*self.get_practice()[u]))[0][:practice_length])
                jaccard.append(len(first_set & second_set) / float(len(first_set | second_set)))
            jaccard_mean, jaccard_std = numpy.mean(jaccard), numpy.std(jaccard)
            result = {'mean': jaccard_mean, 'std': jaccard_std}
            self._jaccard[jaccard_key] = result
        return result['mean'], result['std']

    def intersection(self, practice_length=None):
        if practice_length is None:
            practice_length = self._practice_length
        result = self._intersection.get(practice_length)
        if result is None:
            baseline = self._scenario.optimal_simulator()
            intersection = []
            for u in range(self._scenario.number_of_users()):
                first_set = set(list(zip(*baseline.get_practice()[u]))[0][:practice_length])
                second_set = set(list(zip(*self.get_practice()[u]))[0][:practice_length])
                intersection.append(len(first_set & second_set))
            intersection_mean, intersection_std = numpy.mean(intersection), numpy.std(intersection)
            result = {'mean': intersection_mean, 'std': intersection_std}
            self._intersection[practice_length] = result
        return result['mean'], result['std']

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
        result = self._replay.get(str(model))
        if result is None:
            predicted = []
            actual = []
            model.reset()
            for u in sorted(self._users.keys()):
                for item, _, correct, _ in self.get_practice()[u]:
                    predicted.append(model.predict(u, item))
                    model.update(u, item, correct)
                    actual.append(correct)
            result = rmse(predicted, actual)
            self._replay[str(model)] = result
        return result

    def filename(self, directory):
        return directory + '/' + self.hash()

    def hash(self):
        return hashlib.sha1(str(self).encode()).hexdigest()

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
        if self._directory is None or self._stats_loaded:
            return
        filename = self.filename(self._directory) + '_stats.json'
        if not path.exists(filename):
            return
        with open(filename, 'r') as f:
            to_json = json.loads(f.read())
            self._rmse = convert_dict(to_json['rmse'], int, float)
            self._jaccard = convert_dict(to_json['jaccard'], str, dict)
            self._intersection = convert_dict(to_json['intersection'], int, dict)
            self._replay = convert_dict(to_json['replay'], str, float)
            if 'number_of_answers' in to_json:
                self._number_of_answers = convert_dict(to_json['number_of_answers'], int, int)
            self._stats_loaded = True

    def _save_stats(self, directory):
        to_json = {
            'str': str(self),
            'model': str(self._model),
            'rmse': self._rmse,
            'jaccard': self._jaccard,
            'intersection': self._intersection,
            'replay': self._replay
        }
        if self._number_of_answers is not None:
            to_json['number_of_answers'] = self._number_of_answers
        filename = self.filename(directory) + '_stats.json'
        with open(filename, 'w') as f:
            json.dump(to_json, f)

    def __str__(self):
        return 'simulator, model: %s, practice length: %s, train: %s, target prob: %.2f' % (
            str(self._model), self._practice_length, self._train, self._target_probability)

    def _get_data(self, practice, practice_length):
        return [(u, p[0], p[2]) for u, ps in practice.items() for p in ps[:practice_length]]

    def _simulate(self, storage, practice_length, number_of_users=None, recommend_fun=recommend):
        if len(storage) > 0:
            return
        self._model.reset()
        for u in range(len(self._users)):
            item_ids = set(self._items.keys())
            storage[u] = []
            for p in range(practice_length):
                predictions = dict([(i, self._model.predict(u, i)) for i in item_ids])
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
            flat_practice = [x for xs in list(practice.values()) for x in xs[:practice_length]]
            storage[practice_length] = rmse(list(zip(*flat_practice))[1], list(zip(*flat_practice))[2])
        return storage[practice_length]


