import json
import hashlib
from os import path
import numpy
import random
from .util import convert_dict
from .model import OptimalModel
from .simulator import Simulator


def load_scenario(json_file, name, version):
    with open(json_file, 'r') as f:
        scenarios = json.loads(f.read())['scenarios']
        return Scenario(list(filter(lambda s: s['name'] == name, scenarios))[0], version=version)


class Scenario:

    def __init__(self, config, version=None):
        self._data = {'config': config}
        self._data['config']['version'] = None
        self._data['storage'] = {}
        self._data['train_set'] = {}
        self._data['test_set'] = {}
        self._simulators = {}
        self._optimal_simulator = None
        self._optimal_simulator_saved = False

    def init_simulator(self, directory, model, practice_length=None, target_probability=None):
        if not self._optimal_simulator_saved and self._optimal_simulator is not None:
            self._optimal_simulator.load(self.filename(directory))
        model.reset()
        simulator = Simulator(
            OptimalModel(self.skills(), self.difficulties(), self.clusters()),
            model,
            self,
            practice_length=practice_length,
            target_probability=target_probability)
        simulator_name = str(simulator)
        found_simulator = self._simulators.get(simulator_name)
        if found_simulator is None:
            simulator.load(self.filename(directory))
            self._simulators[simulator_name] = simulator
            simulator.save(self.filename(directory))
            found_simulator = simulator
        return found_simulator

    def optimal_simulator(self):
        if self._optimal_simulator is None:
            optimal_model = OptimalModel(self.skills(), self.difficulties(), self.clusters())
            self._optimal_simulator = Simulator(
                optimal_model,
                optimal_model,
                self)
        return self._optimal_simulator

    def config_hash(self):
        return hashlib.sha1(json.dumps(self._data['config'], sort_keys=True).encode()).hexdigest()

    def parameter(self, simulator_key, parameter_name):
        return self._data['config']['parameters'][simulator_key][parameter_name]

    def practice_length(self):
        return self._data['config']['practice_length']

    def target_probability(self):
        return self._data['config']['target_probability']

    def number_of_clusters(self):
        return len(self._data['config']['skills'])

    def number_of_items(self):
        return self._data['config']['number_of_items']

    def number_of_users(self):
        return self._data['config']['number_of_users']

    def number_of_items_with_wrong_cluster(self):
        return self._data['config']['wrong_clusters']['number_of_items']

    def affected_wrong_clusters(self):
        return self._data['config']['wrong_clusters']['affected_clusters']

    def write(self, key, value):
        self._data['storage'][key] = value

    def read(self, key):
        return self._data['storage'].get(key)

    def load(self, directory):
        if not path.exists(self.filename(directory) + '.json'):
            return
        with open(self.filename(directory) + '.json', 'r') as f:
            self._data = json.loads(f.read())
            for storage_key in ['test_set', 'train_set']:
                storage = self._data[storage_key]
                for key in ['difficulties']:
                    if key in storage:
                        storage[key] = convert_dict(storage[key], int, float)
                for key in ['skills']:
                    if key in storage:
                        storage[key] = convert_dict(storage[key], int, lambda xs: list(map(float, list(xs))))
                for key in ['clusters']:
                    if key in storage:
                        storage[key] = convert_dict(storage[key], int, int)

    def save(self, directory):
        with open(self.filename(directory) + '.json', 'w') as f:
            json.dump(self._data, f)
        for simulator in list(self._simulators.values()):
            simulator.save(self.filename(directory))
        if self._optimal_simulator is not None:
            self._optimal_simulator.save(self.filename(directory))

    def skills(self):
        return self._skills(self._data['test_set'])

    def clusters(self):
        return self._clusters(self._data['test_set'])

    def difficulties(self):
        return self._difficulties(self._data['test_set'])

    def train_skills(self):
        return self._skills(self._data['train_set'])

    def train_clusters(self):
        return self._clusters(self._data['train_set'])

    def train_difficulties(self):
        return self._difficulties(self._data['train_set'])

    def filename(self, directory):
        return directory + '/' + self._data['config']['name'] + '_' + self.config_hash()[:10]

    def _clusters(self, storage):
        if 'clusters' not in storage:
            number_of_items = self.number_of_items()
            number_of_clusters = self.number_of_clusters()
            storage['clusters'] = dict([(i, random.choice(list(range(number_of_clusters)))) for i in range(number_of_items)])
        return storage['clusters']

    def _difficulties(self, storage):
        if 'difficulties' not in storage:
            number_of_items = int(self._data['config']['number_of_items'])
            difficulty_mean = float(self._data['config']['difficulty']['mean'])
            difficulty_std = float(self._data['config']['difficulty']['std'])
            storage['difficulties'] = dict([(i, numpy.random.normal(difficulty_mean, difficulty_std)) for i in range(number_of_items)])
        return storage['difficulties']

    def _skills(self, storage):
        if 'skills' not in storage:
            number_of_users = self.number_of_users()
            def _skill():
                skills = []
                for skill in self._data['config']['skills']:
                    skills.append(numpy.random.normal(float(skill['mean']), float(skill['std'])))
                return skills
            storage['skills'] = dict([(u, _skill()) for u in range(number_of_users)])
        return storage['skills']
