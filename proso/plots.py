from util import window_fun, rmse
import numpy
import seaborn as sns
import pandas
import math
from model import ClusterEloModel, OptimalModel, NoiseModel
from collections import defaultdict


COLORS = sns.color_palette()

def plot_scenario(plot, scenario):
    skills = defaultdict(list)
    difficulties = []
    for _, user_skills in scenario.skills().iteritems():
        for i, skill in enumerate(user_skills):
            skills[i].append(skill)
    for d in scenario.difficulties().itervalues():
        difficulties.append(d)
    subplot = plot.add_subplot(121)
    subplot.hist(difficulties)
    subplot.set_xlabel('Difficulty')
    subplot.set_ylabel('Number of Items')

    subplot = plot.add_subplot(122)
    subplot.hist(skills.values())
    subplot.set_xlabel('Skill')
    subplot.set_ylabel('Number of Users')

    plot.set_size_inches(17, 5)
    return plot


def plot_number_of_answers(plot, scenario, simulators):
    names = []
    numbers = []
    probs = []
    subplot = plot.add_subplot(122)
    for simulator_name, simulator in simulators.iteritems():
        nums = simulator.number_of_answers().values()
        numbers.append(nums)
        practice = map(lambda x: x[3], [x for xs in simulator.get_practice().values() for x in xs])
        probs.append(practice)
        names.append(simulator_name)
        subplot.plot(sorted(nums, reverse=True), label=simulator_name)
    subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    subplot.set_xlabel('Item (sorted according to the number of answers)')
    subplot.set_ylabel('Number of Answers')

    subplot = plot.add_subplot(121)
    subplot.set_xlabel('Difficulty')
    subplot.set_ylabel('Number of Answers')
    subplot.hist(probs, label=names, bins=20)

    plot.set_size_inches(17, 5)
    return plot


def plot_model_parameters(plot, scenario, model_factory, parameter_x, parameter_y):
    subplot = plot.add_subplot(111)
    name_x, min_x, max_x, steps_x = parameter_x
    name_y, min_y, max_y, steps_y = parameter_y
    xs = numpy.linspace(min_x, max_x, steps_x)
    ys = numpy.linspace(min_y, max_y, steps_y)
    rmses = pandas.DataFrame(columns=xs, index=ys, dtype=float)
    for x in xs:
        for y in ys:
            model = model_factory(x, y)
            rmses[x][y] = scenario.optimal_simulator().replay(model)
    img = subplot.pcolor(rmses)
    subplot.set_xlabel(name_x)
    subplot.set_xticklabels(
        map(lambda x: '%.2f' % x, list(numpy.linspace(min_x, max_x, len(subplot.get_xticks()) - 1))) + ['MAX'])
    subplot.set_ylabel(name_y)
    subplot.set_yticklabels(
        map(lambda x: '%.2f' % x, list(numpy.linspace(min_y, max_y, len(subplot.get_yticks()) - 1))) + ['MAX'])
    plot.colorbar(img)
    return plot


def plot_noise_vs_jaccard(plot, scenario, optimal_simulator, destination, std_step=0.05, std_max=1):
    stds = []
    rmses = []
    jaccard = []
    for std in numpy.arange(0, std_max, std_step):
        model = OptimalModel(scenario.skills(), scenario.difficulties(), scenario.clusters(), noise=std)
        simulator = scenario.init_simulator(destination, model)
        stds.append(std)
        rmses.append(simulator.rmse())
        jaccard.append(simulator.jaccard()[0])

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
            ClusterEloModel(scenario, clusters=scenario.clusters(), number_of_items_with_wrong_cluster=wrong_clusters))
        wrong.append(wrong_clusters)
        rmses.append(simulator.rmse())
        jaccard.append(simulator.jaccard()[0])

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
    jaccard_trends = scenario.read('plot_jaccard__trends')
    if jaccard_trends is None:
        intersections_all = []
        jaccard_trends = []
        baseline = simulators['Optimal']
        for simulator_name, simulator in simulators.iteritems():
            if simulator_name == 'Optimal':
                continue
            jaccard = []
            intersections = []
            for i in xrange(1, scenario.practice_length() + 1):
                jaccard.append(simulator.jaccard(i)[0])
            jaccard_trends.append(jaccard)
        scenario.write('plot_jaccard__trends', jaccard_trends)

    subplot = plot.add_subplot(111)
    for i, (simulator_name, _) in enumerate(filter(lambda (n, s): n != 'Optimal', simulators.items())):
        subplot.plot(range(scenario.practice_length()), jaccard_trends[i], label=simulator_name)
    subplot.set_ylabel('Jaccard Similarity Coefficient')
    subplot.set_xlabel('Number of Attempts')
    subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return plot

def plot_rmse_complex(plot, scenario, simulators):
    simulators_rmse = scenario.read('plot_rmse_complex__rmse')
    if simulators_rmse is None:
        simulators_rmse = []
        for simulator_name, simulator in simulators.iteritems():
            current_rmse = []
            for data_name, data_provider in simulators.iteritems():
                current_rmse.append(data_provider.replay(simulator._model))
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
    plot.set_size_inches(17, 5)
    return plot
