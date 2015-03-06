from util import window_fun, rmse
import numpy
import seaborn as sns
import pandas
import math
from model import ClusterEloModel, OptimalModel, NoiseModel


COLORS = sns.color_palette()


def plot_model_parameters(plot, scenario, simulator_factory, parameter_x, parameter_y):
    subplot = plot.add_subplot(111)
    name_x, min_x, max_x, steps_x = parameter_x
    name_y, min_y, max_y, steps_y = parameter_y
    xs = numpy.linspace(min_x, max_x, steps_x)
    ys = numpy.linspace(min_y, max_y, steps_y)
    rmses = pandas.DataFrame(columns=xs, index=ys, dtype=float)
    for x in xs:
        for y in ys:
            simulator = simulator_factory(x, y)
            rmses[x][y] = simulator.rmse()
    img = subplot.pcolor(rmses)
    subplot.set_xlabel(name_y)
    subplot.set_xticklabels(
        list(numpy.linspace(min_x, max_x, len(subplot.get_xticks()) - 1)) + ['MAX'])
    subplot.set_ylabel(name_x)
    subplot.set_yticklabels(
        list(numpy.linspace(min_y, max_y, len(subplot.get_yticks()) - 1)) + ['MAX'])
    plot.colorbar(img)
    return plot


def plot_noise_vs_jaccard(plot, scenario, optimal_simulator, destination, std_step=0.01, std_max=0.35):
    stds = []
    rmses = []
    jaccard = []
    optimal_model = OptimalModel(scenario.skills(), scenario.difficulties(), scenario.clusters())
    for std in numpy.arange(0, std_max, std_step):
        simulator = scenario.init_simulator(destination, NoiseModel(optimal_model, std))
        user_jaccard = []
        for u in xrange(scenario.number_of_users()):
            first_set = set(zip(*optimal_simulator.get_practice()[u])[0])
            second_set = set(zip(*simulator.get_practice()[u])[0])
            user_jaccard.append(len(first_set & second_set) / float(len(first_set | second_set)))
        stds.append(std)
        rmses.append(simulator.rmse())
        jaccard.append(numpy.mean(user_jaccard))

    subplot = plot.add_subplot(121)
    subplot.plot(stds, rmses)
    subplot.set_xlabel('Noise (standard deviation)')
    subplot.set_ylabel('RMSE')

    subplot = plot.add_subplot(122)
    subplot.plot(stds, jaccard)
    subplot.set_xlabel('Noise (standard deviation)')
    subplot.set_ylabel('Jaccard Similarity Coefficient')

    plot.set_size_inches(17, 5)
    return plot


def plot_wrong_clusters_vs_jaccard(plot, scenario, optimal_simulator, destination, step=5):
    rmses = []
    jaccard = []
    wrong = []
    for wrong_clusters in xrange(0, int(math.ceil(len(scenario.difficulties()) / 2.0)) + 1, step):
        simulator = scenario.init_simulator(
            destination,
            ClusterEloModel(clusters=scenario.clusters(), number_of_items_with_wrong_cluster=wrong_clusters))
        user_jaccard = []
        for u in xrange(scenario.number_of_users()):
            first_set = set(zip(*optimal_simulator.get_practice()[u])[0])
            second_set = set(zip(*simulator.get_practice()[u])[0])
            user_jaccard.append(len(first_set & second_set) / float(len(first_set | second_set)))
        wrong.append(wrong_clusters)
        rmses.append(simulator.rmse())
        jaccard.append(numpy.mean(user_jaccard))

    subplot = plot.add_subplot(121)
    subplot.plot(wrong, rmses)
    subplot.set_xlabel('Number of Items with Wrong Cluster')
    subplot.set_ylabel('RMSE')

    subplot = plot.add_subplot(122)
    subplot.plot(wrong, jaccard)
    subplot.set_xlabel('Number of Items with Wrong Cluster')
    subplot.set_ylabel('Jaccard Similarity Coefficient')

    plot.set_size_inches(17, 5)
    return plot



def plot_jaccard(plot, scenario, simulators):
    intersections_all = scenario.read('plot_jaccard__all')
    jaccard_trends = scenario.read('plot_jaccard__trends')
    if intersections_all is None or jaccard_trends is None:
        intersections_all = []
        jaccard_trends = []
        baseline = simulators['Optimal']
        for simulator_name, simulator in simulators.iteritems():
            if simulator_name == 'Optimal':
                continue
            jaccard = []
            intersections = []
            for i in xrange(1, scenario.practice_length() + 1):
                current_jaccard = []
                current_intersections = []
                for u in xrange(scenario.number_of_users()):
                    first_set = set(zip(*baseline.get_practice()[u])[0][:i])
                    second_set = set(zip(*simulator.get_practice()[u])[0][:i])
                    current_jaccard.append(len(first_set & second_set) / float(len(first_set | second_set)))
                    current_intersections.append(len(first_set & second_set))
                jaccard.append(current_jaccard)
                intersections.append(current_intersections)
            intersections_all.append(list(intersections))
            jaccard_trends.append(window_fun(jaccard, numpy.mean, size=1))
        scenario.write('plot_jaccard__all', intersections_all)
        scenario.write('plot_jaccard__trends', jaccard_trends)

    subplot = plot.add_subplot(122)
    for i, (simulator_name, _) in enumerate(filter(lambda (n, s): n != 'Optimal', simulators.items())):
        subplot.plot(range(scenario.practice_length()), jaccard_trends[i], label=simulator_name)
    subplot.set_ylabel('Jaccard Similarity Coefficient')
    subplot.set_xlabel('Number of Attempts')
    subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    subplot = plot.add_subplot(121)
    subplot.hist(map(lambda xs: xs[scenario.practice_length()-1], intersections_all), bins=10)
    subplot.set_xlabel('Number of Items in Itersection of Practice')
    subplot.set_ylabel('Number of Users')

    plot.set_size_inches(17, 5)
    return plot

def plot_rmse_complex(plot, scenario, simulators):
    simulators_rmse = scenario.read('plot_rmse_complex__rmse')
    if simulators_rmse is None:
        simulators_rmse = []
        for simulator_name, simulator in simulators.iteritems():
            current_rmse = []
            for data_name, data_provider in simulators.iteritems():
                rmse.append(data_provider.replay(simulator._model))
            simulators_rmse.append(current_rmse)
        scenario.write('plot_rmse_complex__rmse', simulators_rmse)
    index = numpy.arange(len(simulators))
    bar_width = 0.1
    subplot = plot.add_subplot(111)
    names = map(lambda (n, _): n, simulators.items())
    for i, current_rmse in enumerate(simulators_rmse):
        subplot.bar(index + i * bar_width, current_rmse, bar_width, label=names[i], color=COLORS[i])
    subplot.set_xticklabels(names)
    subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    subplot.set_xlabel('Data Set')
    subplot.set_ylabel('RMSE')
    subplot.set_ylim(0.4, 0.6)
    plot.set_size_inches(17, 5)
    return plot
