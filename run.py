from argparse import ArgumentParser
from os import path, makedirs
import proso.scenario
from proso.model import *
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
    parser.add_argument(
        '--skip-groups',
        dest='skip_groups',
        nargs='+',
        type=str)
    parser.add_argument(
        '--skip-cache',
        action='store_true',
        dest='skip_cache',
        help='skip saving the cache')
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

    models = {
        'Elo': ClusterEloModel(scenario, clusters={}),
        'Elo, Concepts': ClusterEloModel(scenario, clusters=clusters),
        'Elo, Concepts (wrong)': ClusterEloModel(scenario, clusters=clusters, number_of_items_with_wrong_cluster=scenario.number_of_items_with_wrong_cluster()),
        'Naive': NaiveModel(),
        'Constant': ConstantModel(constant=scenario.target_probability())
    }
    simulators = {
        'Optimal': scenario.optimal_simulator(),
        'Elo': scenario.init_simulator(args.destination, ClusterEloModel(scenario, clusters={})),
        'Elo, Concepts': scenario.init_simulator(args.destination, ClusterEloModel(scenario, clusters=clusters)),
        'Elo, Concepts (wrong)': scenario.init_simulator(args.destination, ClusterEloModel(scenario, clusters=clusters, number_of_items_with_wrong_cluster=scenario.number_of_items_with_wrong_cluster())),
        'Naive': scenario.init_simulator(args.destination, NaiveModel()),
        'Constant': scenario.init_simulator(args.destination, ConstantModel(constant=scenario.target_probability()))
    }
    if args.skip_groups is None or 'common' not in args.skip_groups:
        fig = plt.figure()
        savefig(args, scenario, plot_jaccard(fig, scenario, simulators), 'jaccard')
        fig = plt.figure()
        savefig(args, scenario, plot_rmse_complex(fig, scenario, simulators), 'rmse_complex')
        fig = plt.figure()
        savefig(args, scenario, plot_number_of_answers_per_difficulty(fig, scenario, simulators), 'number_of_answers')
        #fig = plt.figure()
        #savefig(args, scenario, plot_scenario(fig, scenario), 'scenario')
    #if args.skip_groups is None or 'target_prob' not in args.skip_groups:
        #fig = plt.figure()
        #savefig(args, scenario, plot_target_probability_vs_jaccard_rmse(fig, scenario, args.destination, models), 'target_prob_vs_jaccard_rmse')
    if args.skip_groups is None or 'noise' not in args.skip_groups:
        #fig = plt.figure()
        #savefig(args, scenario, plot_wrong_clusters_vs_jaccard(fig, scenario, simulators['Optimal'], args.destination), 'wrong_clusters_vs_jaccard')
        fig = plt.figure()
        plot_noise_vs_jaccard_number_of_answers(fig.add_subplot(121), scenario, simulators['Optimal'], args.destination)
        plot_number_of_answers_distribution(fig.add_subplot(122), scenario, simulators)
        fig.set_size_inches(17, 5)
        savefig(args, scenario, fig, 'noise_vs_jaccard_number_of_answers')
    #if args.skip_groups is None or 'fitting' not in args.skip_groups:
        #fig = plt.figure()
        #savefig(args, scenario, plot_model_parameters(
            #fig,
            #scenario,
            #lambda x, y: ClusterEloModel(scenario, clusters={}, alpha=x, dynamic_alpha=y),
            #('alpha', 0.25, 1.5, 6),
            #('beta', 0.01, 0.1, 10)
        #), 'elo_parameters')
        #fig = plt.figure()
        #savefig(args, scenario, plot_model_parameters(
            #fig,
            #scenario,
            #lambda x, y: ClusterEloModel(scenario, clusters=clusters, alpha=x, dynamic_alpha=y),
            #('alpha', 0.25, 1.5, 6),
            #('beta', 0.01, 0.1, 10)
        #), 'elo_clusters_parameters')
    if not args.skip_cache:
        scenario.save(args.destination)


if __name__ == "__main__":
    main()
