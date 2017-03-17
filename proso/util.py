from sklearn.metrics import mean_squared_error
import math


def running_fun(xs, fun):
    result = []
    visited = []
    for i, x in enumerate(xs):
        visited.append(x)
        result.append(fun(visited))
    return result


def window_fun(xs, fun, size):
    return running_fun(xs, lambda running: fun(running[-size:]))


def rmse(xs, ys):
    return math.sqrt(mean_squared_error(xs, ys))


def convert_dict(json_dict, key_type, value_type):
    return dict([(key_type(k_v[0]), value_type(k_v[1])) for k_v in list(json_dict.items())])
