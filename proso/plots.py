from util import window_fun, rmse
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
    return plot


def plot_rmse(plot, scenario, simulators):
    rmses = scenario.read('plot_rmse__ones')
    rmses_all = scenario.read('plot_rmse__all')
    if rmse is None or rmses_all is None:
        rmses = []
        rmses_all = []
        for simulator_name, simulator in simulators.iteritems():
            rmses.append(simulator.rmse(scenario.practice_length()))
            rmses_all.append(simulator.rmse_all(scenario.number_of_items()))
        scenario.write('plot_rmse__ones', rmses)
        scenario.write('plot_rmse__all', rmses_all)
    names = map(lambda (n, _): n, simulators.items())
    index = numpy.arange(len(simulators))
    bar_width = 0.35
    subplot = plot.add_subplot(111)
    subplot.set_xticklabels(names)
    subplot.bar(index, rmses, bar_width, label='only a limited recommended practice', color=COLORS[0])
    subplot.bar(index + bar_width, rmses_all, bar_width, label='all', color=COLORS[1])
    subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    subplot.set_xlabel('Model')
    subplot.set_ylabel('RMSE')
    subplot.set_ylim(0.4, 0.6)
    return plot

def plot_rmse_complex(plot, scenario, simulators):
    simulators_rmse = scenario.read('plot_rmse_complex__rmse')
    if simulators_rmse is None:
        simulators_rmse = []
        for simulator_name, simulator in simulators.iteritems():
            rmse = []
            for data_name, data_provider in simulators.iteritems():
                rmse.append(data_provider.replay(simulator._model))
            simulators_rmse.append(rmse)
        scenario.write('plot_rmse_complex__rmse', simulators_rmse)
    index = numpy.arange(len(simulators))
    bar_width = 0.1
    subplot = plot.add_subplot(111)
    names = map(lambda (n, _): n, simulators.items())
    for i, rmse in enumerate(simulators_rmse):
        subplot.bar(index + i * bar_width, rmse, bar_width, label=names[i], color=COLORS[i])
    subplot.set_xticklabels(names)
    subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    subplot.set_xlabel('Data Set')
    subplot.set_ylabel('RMSE')
    subplot.set_ylim(0.4, 0.6)
    plot.set_size_inches(17, 5)
    return plot
