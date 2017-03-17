from argparse import ArgumentParser
from os import path, makedirs
import proso.scenario
from proso.model import ClusterEloModel, NaiveModel, ConstantModel
from proso.plots import plot_intersection, plot_rmse_complex, plot_number_of_answers_per_difficulty, plot_noise_vs_intersection_number_of_answers, plot_number_of_answers_distribution
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
        default='svg',
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


def savefig(args, scenario, name):
    if not path.exists(scenario.filename(args.destination)):
        makedirs(scenario.filename(args.destination))
    filename = scenario.filename(args.destination) + '/' + name + '.' + args.output
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(' -- saving', filename)
    plt.close()


def main():
    args = parser_init().parse_args()
    if not path.exists(args.destination):
        makedirs(args.destination)
    scenario = proso.scenario.load_scenario(args.settings, args.name, VERSION)
    scenario.load(args.destination)

    clusters = scenario.clusters()

    simulators = {
        'Optimal': scenario.optimal_simulator(),
        'Elo': scenario.init_simulator(args.destination, ClusterEloModel(scenario, clusters={})),
        'Elo, Concepts': scenario.init_simulator(args.destination, ClusterEloModel(scenario, clusters=clusters)),
        'Elo, Concepts (wrong)': scenario.init_simulator(args.destination, ClusterEloModel(scenario, clusters=clusters, number_of_items_with_wrong_cluster=scenario.number_of_items_with_wrong_cluster())),
        'Naive': scenario.init_simulator(args.destination, NaiveModel()),
        'Constant': scenario.init_simulator(args.destination, ConstantModel(constant=scenario.target_probability()))
    }
    if args.skip_groups is None or 'common' not in args.skip_groups:
        plot_intersection(scenario, simulators)
        savefig(args, scenario, 'intersection')
        plt.gcf().set_size_inches(14, 4)
        plot_rmse_complex(scenario, simulators)
        savefig(args, scenario, 'rmse_complex')
        plt.gcf().set_size_inches(14, 4)
        plot_number_of_answers_per_difficulty(scenario, simulators),
        savefig(args, scenario, 'number_of_answers')

    if args.skip_groups is None or 'noise' not in args.skip_groups:
        plt.gcf().set_size_inches(14, 4)
        plt.subplot(121)
        plot_noise_vs_intersection_number_of_answers(scenario, simulators['Optimal'], args.destination)
        plt.subplot(122)
        plot_number_of_answers_distribution(scenario, simulators)
        savefig(args, scenario, 'noise_vs_intersection_number_of_answers')
    if not args.skip_cache:
        scenario.save(args.destination)


if __name__ == "__main__":
    main()
