from argparse import ArgumentParser
from os import path, makedirs
import proso.scenario
from proso.model import *
from proso.simulator import Simulator
import random
from proso.plots import *
import matplotlib.pyplot as plt


VERSION = 1


def parser_init():
    parser = ArgumentParser()
    parser.add_argument(
        '-s',
        '--settings',
        metavar='FILE',
        required=True,
        help='path to the JSON file with settings')
    parser.add_argument(
        '-n',
        '--name',
        metavar='SCENARIO',
        required=True,
        help='name of the scenario to be executed')
    parser.add_argument(
        '-d',
        '--destination',
        metavar='DIR',
        required=True,
        help='path to the directory where the created data will be saved')
    parser.add_argument(
        '-o',
        '--output',
        metavar='EXT',
        dest='output',
        default='png',
        help='extension for the output fles')
    return parser


def savefig(args, scenario, figure, name):
    if not path.exists(scenario.filename(args.destination)):
        makedirs(scenario.filename(args.destination))
    filename = scenario.filename(args.destination) + '/' + name + '.' + args.output
    figure.tight_layout()
    figure.savefig(filename, bbox_inches='tight')
    print ' -- saving', filename
    plt.close(figure)


def main():
    args = parser_init().parse_args()
    if not path.exists(args.destination):
        makedirs(args.destination)
    scenario = proso.scenario.load_scenario(args.settings, args.name, VERSION)
    scenario.load(args.destination)

    users = scenario.skills()
    items = scenario.difficulties()
    clusters = scenario.clusters()

    optimal_model = OptimalModel(users, items, clusters)
    optimal_simulator = Simulator(
        optimal_model,
        optimal_model,
        users,
        items,
        clusters,
        practice_length=scenario.practice_length())

    elo_simulator = Simulator(
        optimal_model,
        ClusterEloModel(clusters={}),
        users,
        items,
        clusters,
        practice_length=scenario.practice_length())

    cluster_simulator = Simulator(
        optimal_model,
        ClusterEloModel(clusters=clusters),
        users,
        items,
        clusters,
        practice_length=scenario.practice_length())

    wrong_items = random.sample(items.keys(), scenario.number_of_items_with_wrong_cluster())
    wrong_clusters = dict(clusters.items())
    for i in wrong_items:
        available_clusters = set(range(scenario.number_of_clusters()))
        available_clusters.remove(wrong_clusters[i])
        wrong_clusters[i] = random.choice(list(available_clusters))
    wrong_cluster_simulator = Simulator(
        optimal_model,
        ClusterEloModel(clusters=wrong_clusters),
        users,
        items,
        clusters,
        practice_length=scenario.practice_length())

    naive_simulator = Simulator(
        optimal_model,
        NaiveModel(),
        users,
        items,
        clusters,
        practice_length=scenario.practice_length())

    const_simulator = Simulator(
        optimal_model,
        ConstantModel(constant=scenario.target_probability()),
        users,
        items,
        clusters,
        practice_length=scenario.practice_length())

    simulators = {
        'Elo': elo_simulator,
        'Elo, Clusters': cluster_simulator,
        'Elo, Clusters (wrong)': wrong_cluster_simulator,
        'Naive': naive_simulator,
        'Constant': const_simulator,
        'Optimal': optimal_simulator
    }

    simulators_cache = scenario.read('simulators')
    if simulators_cache is None:
        simulators_cache = {}
    for name, simulator in simulators.iteritems():
        print ' -- ', name, 'simulating'
        if name not in simulators_cache:
            simulator.simulate()
            simulator.simulate_all()
            simulators_cache[name] = simulator.to_json()
        else:
            simulator.load_json(simulators_cache[name])
    scenario.write('simulators', simulators_cache)
    fig = plt.figure()
    plot_intersections(fig, scenario, simulators)
    savefig(args, scenario, fig, 'intersections')
    scenario.save(args.destination)


if __name__ == "__main__":
    main()
