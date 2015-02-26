from util import window_fun
import numpy
import seaborn as sns


COLORS = sns.color_palette()


def plot_intersections(plot, scenario, simulators):
    intersections_all = scenario.read('plot_intersections__all')
    intersections_trends = scenario.read('plot_intersections__trends')
    if intersections_all is None or intersections_trends is None:
        intersections_all = []
        intersections_trends = []
        baseline = simulators['Optimal']
        for simulator_name, simulator in simulators.iteritems():
            if simulator_name == 'Optimal':
                continue
            intersections = []
            for i in xrange(1, scenario.practice_length() + 1):
                current_intersections = []
                for u in xrange(scenario.number_of_users()):
                    current_intersections.append(len(
                        set(zip(*baseline._practice[u])[0][:i]) & set(zip(*simulator._practice[u])[0][:i])
                    ))
                intersections.append(current_intersections)
            intersections_all.append(list(intersections))
            intersections_trends.append(window_fun(intersections, numpy.mean, size=1))
        scenario.write('plot_intersections__all', intersections_all)
        scenario.write('plot_intersections__trends', intersections_trends)

    subplot = plot.add_subplot(122)
    for i, (simulator_name, _) in enumerate(filter(lambda (n, s): n != 'Optimal', simulators.items())):
        subplot.plot(range(scenario.practice_length()), intersections_trends[i], label=simulator_name)
    subplot.set_ylabel('Number of Items in Itersection of Practice')
    subplot.set_xlabel('Number of Attempts')
    subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    subplot = plot.add_subplot(121)
    subplot.hist(map(lambda xs: xs[scenario.practice_length()-1], intersections_all), bins=10)
    subplot.set_xlabel('Number of Items in Itersection of Practice')
    subplot.set_ylabel('Number of Users')

    plot.set_size_inches(17, 5)
