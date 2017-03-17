from .util import window_fun, rmse
import numpy
import seaborn as sns
import pandas
import math
from .model import ClusterEloModel, OptimalModel, NoiseModel
from collections import defaultdict
from .simulator import prediction_score

SNS_STYLE = {'style': 'white', 'font_scale':1.7}

sns.set(**SNS_STYLE)


COLORS = sns.color_palette()

def plot_scenario(plot, scenario):
    skills = defaultdict(list)
    difficulties = []
    for _, user_skills in scenario.skills().items():
        for i, skill in enumerate(user_skills):
            skills[i].append(skill)
    for d in scenario.difficulties().values():
        difficulties.append(d)
    subplot = plot.add_subplot(121)
    subplot.hist(difficulties)
    subplot.set_xlabel('Difficulty')
    subplot.set_ylabel('Number of Items')

    subplot = plot.add_subplot(122)
    subplot.hist(list(skills.values()))
    subplot.set_xlabel('Skill')
    subplot.set_ylabel('Number of Users')

    plot.set_size_inches(17, 5)
    return plot


def plot_target_probability_vs_jaccard_rmse(plot, scenario, destination, models, target_prob_step=0.05, target_prob_min=0.5, target_prob_max=1):
    subplot = plot.add_subplot(121)
    optimal_model = OptimalModel(scenario.skills(), scenario.difficulties(), scenario.clusters())
    target_probs = numpy.arange(target_prob_min, target_prob_max, target_prob_step)
    for model_name, model in models.items():
        model_jaccards = []
        for target_prob in target_probs:
            simulator = scenario.init_simulator(destination, model, target_probability=target_prob)
            optimal_simulator = scenario.init_simulator(destination, optimal_model, target_probability=target_prob)
            model_jaccards.append(simulator.jaccard(baseline=optimal_simulator)[0])
        subplot.plot(target_probs, model_jaccards, label=model_name)
    subplot.set_xlabel('Target Difficulty')
    subplot.set_ylabel('Jaccard Similarity Coefficient')
    subplot = plot.add_subplot(122)
    for model_name, model in models.items():
        model_rmses = []
        for target_prob in target_probs:
            rmse = scenario.init_simulator(destination, optimal_model, target_probability=target_prob).replay(model)
            model_rmses.append(rmse)
        subplot.plot(target_probs, model_rmses, label=model_name)
    subplot.set_xlabel('Target Difficulty')
    subplot.set_ylabel('RMSE')
    subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plot.set_size_inches(17, 5)
    return plot


def plot_number_of_answers_distribution(subplot, scenario, simulators, bins=20):
    names = []
    numbers = []
    for simulator_name, simulator in simulators.items():
        nums = list(simulator.number_of_answers().values())
        numbers.append(nums)
        names.append(simulator_name)
        subplot.plot(sorted(nums, reverse=True), label=simulator_name, lw='2')
    subplot.set_xlabel('Item (sorted according to the number of answers)')
    subplot.set_ylabel('Number of Answers')
    subplot.legend(loc='upper right')
    return subplot


def plot_number_of_answers_per_difficulty(plot, scenario, simulators, bins=20):
    names = []
    probs = []
    for simulator_name, simulator in simulators.items():
        practice = [x[3] for x in [x for xs in list(simulator.get_practice().values()) for x in xs]]
        probs.append(practice)
        names.append(simulator_name)
    target_probs = numpy.linspace(0, 1, bins)
    subplot = plot.add_subplot(111)
    subplot.set_xlabel('True Probability of Correct Answer')
    subplot.set_ylabel('Number of Answers')
    subplot.hist(probs, label=names, bins=bins)
    subplot_twin = subplot.twinx()
    subplot_twin.xaxis.grid(False)
    subplot_twin.yaxis.grid(False)
    subplot_twin.plot(
        target_probs,
        [prediction_score(x, scenario.target_probability()) for x in target_probs],
        '--',
        lw=1,
        color='gray',
        alpha=0.5,
        label='Score')
    subplot_twin.set_ylabel('Score')
    subplot.legend(loc='upper left')
    subplot_twin.legend(loc='upper right')
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
        ['%.2f' % x for x in list(numpy.linspace(min_x, max_x, len(subplot.get_xticks()) - 1))] + ['MAX'])
    subplot.set_ylabel(name_y)
    subplot.set_yticklabels(
        ['%.2f' % x for x in list(numpy.linspace(min_y, max_y, len(subplot.get_yticks()) - 1))] + ['MAX'])
    plot.colorbar(img)
    return plot


def plot_noise_vs_intersection_number_of_answers(subplot, scenario, optimal_simulator, destination, std_step=0.01, std_max=0.35):
    stds = []
    rmses = []
    intersection = []
    for std in numpy.arange(0, std_max, std_step):
        model = OptimalModel(scenario.skills(), scenario.difficulties(), scenario.clusters(), noise=std)
        simulator = scenario.init_simulator(destination, model)
        stds.append(std)
        rmses.append(simulator.rmse())
        intersection.append(simulator.intersection()[0])

    subplot_twin = subplot.twinx()

    subplot_twin.plot(stds, rmses, '-s', color=COLORS[0], label="RMSE")
    subplot_twin.set_xlabel('Noise (standard deviation)')
    subplot_twin.set_ylabel('RMSE')

    subplot.plot(stds, intersection, '-o', color=COLORS[2], label="Size of the Intersection\nwith the Optimal Practiced Set")
    subplot.set_xlabel('Noise (standard deviation)')
    subplot.set_ylabel('Size of the Intersection')

    subplot.legend(loc="upper center")
    subplot_twin.legend(loc="lower center")

    return subplot


def plot_wrong_clusters_vs_jaccard(plot, scenario, optimal_simulator, destination, step=5):
    rmses = []
    jaccard = []
    wrong = []
    for wrong_clusters in range(0, int(math.ceil(len(scenario.difficulties()) / 2.0)) + 1, step):
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


def plot_intersection(plot, scenario, simulators):
    intersection_trends = scenario.read('plot_intersection__trends')
    if intersection_trends is None:
        intersection_trends = []
        for simulator_name, simulator in simulators.items():
            if simulator_name == 'Optimal':
                continue
            intersection = []
            for i in range(1, scenario.practice_length() + 1):
                intersection.append(simulator.intersection(i)[0])
            intersection_trends.append(intersection)
            print(simulator_name, intersection[-1])
        scenario.write('plot_intersection__trends', intersection_trends)

    subplot = plot.add_subplot(111)
    for i, (simulator_name, _) in enumerate([n_s for n_s in list(simulators.items()) if n_s[0] != 'Optimal']):
        subplot.plot(list(range(scenario.practice_length())), intersection_trends[i], label=simulator_name)
    subplot.set_ylabel('Size of the Intersection')
    subplot.set_xlabel('Number of Attempts')
    subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return plot

def plot_rmse_complex(plot, scenario, simulators):
    simulators_rmse = scenario.read('plot_rmse_complex__rmse')
    if simulators_rmse is None:
        simulators_rmse = []
        for simulator_name, simulator in simulators.items():
            current_rmse = []
            for data_name, data_provider in simulators.items():
                current_rmse.append(data_provider.replay(simulator._model))
            simulators_rmse.append(current_rmse)
        scenario.write('plot_rmse_complex__rmse', simulators_rmse)
    index = numpy.arange(len(simulators))
    bar_width = 0.1
    subplot = plot.add_subplot(111)
    names = [n__[0] for n__ in list(simulators.items())]
    for i, current_rmse in enumerate(simulators_rmse):
        subplot.bar(index + i * bar_width, current_rmse, bar_width, label=names[i], color=COLORS[i])
    subplot.set_xticklabels(names)
    subplot.legend(loc='upper right')
    subplot.set_xlabel('Data Set')
    subplot.xaxis.grid(False)
    subplot.set_ylabel('RMSE')
    subplot.set_ylim(0.4, 0.6)
    plot.set_size_inches(17, 5)
    return plot
