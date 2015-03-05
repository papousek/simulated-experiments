import abc
import math
import random


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

    def __init__(self, clusters, alpha=None, dynamic_alpha=None, number_of_items_with_wrong_cluster=0):
        if alpha is None or dynamic_alpha is None:
            if len(clusters) > 0:
                alpha = 0.3
                dynamic_alpha = 0.04
            else:
                alpha = 0.4
                dynamic_alpha = 0.05
        if number_of_items_with_wrong_cluster > 0:
            wrong_items = random.sample(clusters.keys(), number_of_items_with_wrong_cluster)
            clusters = dict(clusters.items())
            number_of_clusters = len(set(clusters.values()))
            for i in wrong_items:
                available_clusters = set(range(number_of_clusters))
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

