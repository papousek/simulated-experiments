import abc
import math
import random
from collections import defaultdict


def predict(skill):
    return 1.0 / (1 + math.exp(-skill))


class Model:

    @abc.abstractmethod
    def predict(self, user, item):
        pass

    @abc.abstractmethod
    def update(self, user, item, correct):
        pass

    @abc.abstractmethod
    def reset():
        pass


class OptimalModel(Model):

    def __init__(self, users, items, clusters):
        self._users = users
        self._items = items
        self._clusters = clusters

    def predict(self, user, item):
        cluster = self._clusters[item]
        return predict(self._users[user][cluster] - self._items[item])

    def update(self, user, item, correct):
        pass

    def reset(self):
        pass

    def __str__(self):
        return 'optimal'


class ClusterEloModel(Model):

    def __init__(self, scenario, clusters, alpha=None, dynamic_alpha=None, number_of_items_with_wrong_cluster=0, affected_wrong_clusters=None):
        if alpha is None or dynamic_alpha is None:
            if len(clusters) > 0:
                alpha = scenario.parameter('elo_clusters', 'alpha')
                dynamic_alpha = scenario.parameter('elo_clusters', 'beta')
            else:
                alpha = scenario.parameter('elo', 'alpha')
                dynamic_alpha = scenario.parameter('elo', 'beta')
        if number_of_items_with_wrong_cluster > 0:
            if affected_wrong_clusters is None:
                affected_wrong_clusters = scenario.affected_wrong_clusters()
            wrong_items = random.sample(clusters.keys(), number_of_items_with_wrong_cluster)
            clusters = dict(clusters.items())
            for i in wrong_items:
                available_clusters = set(affected_wrong_clusters)
                available_clusters.remove(clusters[i])
                clusters[i] = random.choice(list(available_clusters))
        self._users = {}
        self._items = {}
        self._clusters = clusters
        self._answers = {}
        self._alpha = alpha
        self._dynamic_alpha = dynamic_alpha
        self._number_of_items_with_wrong_cluster = number_of_items_with_wrong_cluster

    def predict(self, user, item):
        cluster = self._clusters.get(item, 0)
        return predict(self._users.get((user, cluster), 0) - self._items.get(item, 0))

    def update(self, user, item, correct):
        cluster = self._clusters.get(item, 0)
        prediction = self.predict(user, item)
        user_nums = self._answers.get(('user', user, cluster), 0)
        item_nums = self._answers.get(('item', item), 0)
        self._users[user, cluster] = self._users.get((user, cluster), 0) + self.alpha(user_nums) * (correct - prediction)
        self._items[item] = self._items.get(item, 0) - self.alpha(item_nums) * (correct - prediction)
        self._answers['user', user, cluster] = self._answers.get(('user', user), 0) + 1
        self._answers['item', item] = self._answers.get(('item', item), 0) + 1

    def alpha(self, n):
        return self._alpha / (1 + self._dynamic_alpha * n)

    def reset(self):
        self._users = {}
        self._items = {}
        self._answers = {}

    def __str__(self):
        result = 'elo cluster (alpha: %.2f, dynamic_alpha: %.2f, wrong: %s)' % (
            self._alpha, self._dynamic_alpha, self._number_of_items_with_wrong_cluster)
        if len(self._clusters) > 0:
            result += ', clusters: %s' % len(self._clusters)
        return result


class NaiveModel(Model):

    def __init__(self):
        self._means = {}
        self._nums = {}

    def predict(self, user, item):
        return self._means.get(item, 0.5)

    def update(self, user, item, correct):
        item_mean = self._means.get(item, 0)
        item_num = self._nums.get(item, 0) + 1
        self._nums[item] = item_num
        self._means[item] = float((item_num - 1) * item_mean + correct) / item_num

    def reset(self):
        self._means = {}
        self._nums = {}

    def __str__(self):
        return 'naive'


class ConstantModel(Model):

    def __init__(self, constant):
        self._constant = constant

    def predict(self, user, item):
        return self._constant

    def update(self, user, item, correct):
        pass

    def reset(self):
        pass

    def __str__(self):
        return 'constant (%.2f)' % self._constant


class NoiseModel(Model):

    def __init__(self, model, std):
        self._model = model
        self._std = std
        self._noise = defaultdict(lambda: random.gauss(0, self._std))

    def predict(self, user, item):
        return self._model.predict(user, item) + self._noise[user, item]

    def update(self, user, item, correct):
        return self._model.update(user, item, correct)

    def reset(self):
        return self._model.reset()

    def __str__(self):
        return 'permanent noise (std: %s): %s' % (self._std, str(self._model))
