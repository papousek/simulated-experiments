from .model import OptimalModel
from .simulator import prediction_score
from collections import defaultdict
import numpy
import matplotlib.pyplot as plt
import pandas
import seaborn as sns

SNS_STYLE = {'style': 'white', 'font_scale': 1.3}

sns.set(**SNS_STYLE)


COLORS = sns.color_palette()


def plot_scenario(scenario):
    skills = defaultdict(list)
    difficulties = []
    for _, user_skills in scenario.skills().items():
        for i, skill in enumerate(user_skills):
            skills[i].append(skill)
    for d in scenario.difficulties().values():
        difficulties.append(d)
    subplot = plt.subplot(121)
    subplot.hist(difficulties)
    subplot.set_xlabel('Difficulty')
    subplot.set_ylabel('Number of Items')

    subplot = plt.subplot(122)
    subplot.hist(list(skills.values()))
    subplot.set_xlabel('Skill')
    subplot.set_ylabel('Number of Users')


def plot_number_of_answers_distribution(scenario, simulators, bins=20):
    names = []
    numbers = []
    for simulator_name, simulator in sorted(simulators.items()):
        nums = list(simulator.number_of_answers().values())
        numbers.append(nums)
        names.append(simulator_name)
        plt.plot(sorted(nums, reverse=True), label=simulator_name, lw='4')
    plt.xlabel('Item (sorted according to the number of answers)')
    plt.ylabel('Number of answers')
    plt.legend(loc='upper right')


def plot_number_of_answers_per_difficulty(scenario, simulators, bins=10):
    names = []
    probs = []
    for simulator_name, simulator in sorted(simulators.items()):
        practice = [x[3] for x in [x for xs in list(simulator.get_practice().values()) for x in xs]]
        probs.append(practice)
        names.append(simulator_name)
    subplot = plt.subplot(111)
    subplot.set_xlabel('True Probability of Correct Answer')
    subplot.set_ylabel('Number of Answers')
    subplot.hist(probs, label=names, bins=bins)
    subplot_twin = subplot.twinx()
    subplot_twin.xaxis.grid(False)
    subplot_twin.yaxis.grid(False)
    xs = numpy.linspace(0, 1, 100)
    subplot_twin.plot(
        xs,
        [prediction_score(x, scenario.target_probability()) for x in xs],
        '--',
        lw=3,
        color='gray',
        label='Score')
    subplot_twin.set_ylabel('Score')
    subplot.legend(loc='upper left')
    subplot_twin.legend(loc='upper right')


def plot_noise_vs_intersection_number_of_answers(scenario, optimal_simulator, destination, std_step=0.01, std_max=0.35):
    stds = []
    rmses = []
    intersection = []
    for std in numpy.arange(0, std_max, std_step):
        model = OptimalModel(scenario.skills(), scenario.difficulties(), scenario.clusters(), noise=std)
        simulator = scenario.init_simulator(destination, model)
        stds.append(std)
        rmses.append(simulator.rmse())
        intersection.append(simulator.intersection()[0])

    plt.plot(stds, intersection, '-o', color=COLORS[2], label="Size of the intersection\nwith the optimal practiced set", lw=3)
    plt.xlabel('Noise (standard deviation)')
    plt.ylabel('Size of the intersection')
    plt.legend(loc="upper center")

    subplot_twin = plt.twinx()
    subplot_twin.plot(stds, rmses, '-s', color=COLORS[0], label="RMSE", lw=3)
    subplot_twin.set_xlabel('Noise (standard deviation)')
    subplot_twin.set_ylabel('RMSE')
    subplot_twin.legend(loc="lower center")


def plot_intersection(scenario, simulators):
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

    for i, (simulator_name, _) in enumerate([n_s for n_s in list(simulators.items()) if n_s[0] != 'Optimal']):
        plt.plot(list(range(scenario.practice_length())), intersection_trends[i], label=simulator_name, linewidth=2)
    plt.ylabel('Size of the Intersection')
    plt.xlabel('Number of Attempts')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_rmse_complex(scenario, simulators):
    simulators_rmse = {}
    for simulator_name, simulator in simulators.items():
        current_rmse = {}
        for data_name, data_provider in simulators.items():
            current_rmse[data_name] = data_provider.replay(simulator._model)
        simulators_rmse[simulator_name] = current_rmse
    scenario.write('plot_rmse_complex__rmse', simulators_rmse)
    to_plot = pandas.DataFrame([{'Model': s, 'Data set': d, 'RMSE': rmse} for (s, s_data) in simulators_rmse.items() for d, rmse in s_data.items()]).sort_values(by=['Model', 'Data set'])
    sns.barplot(x='Data set', y='RMSE', hue='Model', data=to_plot)
    plt.ylabel('RMSE')
    plt.ylim(0.4, 0.6)
    plt.legend(loc='upper center', ncol=2)
