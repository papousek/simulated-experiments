import abc
import math


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


class ClusterEloModel(Model):

    def __init__(self, clusters, alpha=0.8, dynamic_alpha=0.05):
        self._users = {}
        self._items = {}
        self._clusters = clusters
        self._answers = {}
        self._alpha = alpha
        self._dynamic_alpha = dynamic_alpha

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
        self._answers['user', user, cluster] = self._answers.get(('user', user), 0)
        self._answers['item', item] = self._answers.get(('item', item), 0)

    def alpha(self, n):
        return self._alpha / (1 + self._dynamic_alpha * n)

    def reset(self):
        self._users = {}
        self._items = {}
        self._answers = {}


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


class ConstantModel(Model):

    def __init__(self, constant):
        self._constant = constant

    def predict(self, user, item):
        return self._constant

    def update(self, user, item, correct):
        pass

    def reset(self):
        pass

