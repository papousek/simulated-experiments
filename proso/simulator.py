import random
import numpy
from util import rmse


TARGET_PROBABILITY = 0.75


def prediction_score(probability, target_probability=TARGET_PROBABILITY):
    diff = target_probability - probability
    sign = 1 if diff > 0 else -1
    normed_diff = abs(diff) / max(0.001, abs(target_probability - 0.5 + sign * 0.5))
    return 1 - normed_diff


def recommend(items_with_predictions):
    scored = map(
        lambda (i, p): (prediction_score(p), random.random(), i),
        items_with_predictions.items())
    return sorted(scored, reverse=True)[0][2]


def recommend_random(items_with_predictions):
    return random.choice(items_with_predictions.keys())


class Simulator:

    def __init__(self, optimal_model, model, users, items, clusters, practice_length):
        self._optimal_model = optimal_model
        self._model = model
        self._users = users
        self._items = items
        self._clusters = clusters
        self._practice = {}
        self._practice_all = {}
        self._practice_length = practice_length
        self._rmse = {}
        self._rmse_all = {}

    def simulate(self):
        self._simulate(
            self._practice,
            self._practice_length,
            recommend_fun=recommend)

    def simulate_all(self):
        self._simulate(
            self._practice_all,
            len(self._items),
            recommend_fun=recommend_random)

    def _simulate(self, storage, practice_length, number_of_users=None, recommend_fun=recommend):
        if len(storage) > 0:
            raise Exception('Simulation has been already executed.')
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

    def rmse(self, practice_length):
        return self._compute_rmse(self._rmse, self._practice, practice_length)

    def rmse_all(self, practice_length):
        return self._compute_rmse(self._rmse_all, self._practice_all, practice_length)

    def _compute_rmse(self, storage, practice, practice_length):
        if storage.get(practice_length) is None:
            flat_practice = [x for xs in practice.values() for x in xs[:practice_length]]
            storage[practice_length] = rmse(zip(*flat_practice)[1], zip(*flat_practice)[2])
        return storage[practice_length]

    def get_data(self, practice_length):
        return self._get_data(self._practice, practice_length)

    def get_data_all(self, practice_length):
        return self._get_data(self._practice_all, practice_length)

    def replay(self, model):
        predicted = []
        actual = []
        model.reset()
        for u in sorted(self._users.keys()):
            for item, _, correct, _ in self._users[u]:
                predicted.append(model.predict(u, item))
                model.update(u, item, correct)
                actual.append(correct)
        return rmse(predicted, actual)

    def _get_data(self, practice, practice_length):
        [(u, p[0], p[2]) for u, ps in practice.iteritems() for p in ps[:practice_length]]
