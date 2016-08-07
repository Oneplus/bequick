#!/usr/bin/env python
def flush_parameters(params, now):
    """
    At the end of each iteration, flash the parameters

    Parameters
    ----------
    params : dict
        The parameters
    now : int
        The current time
    """
    for feature in params:
        w = params[feature]
        w[2] += (now - w[1]) * w[0]
        w[1] = now


def update_parameters(features, params, now, scale):
    """
    update parameter with features

    Parameters
    ----------
    features: list(str)
    params: dict
    now: int, the timestep
    scale: float, the scale

    """
    for feature in features:
        if feature not in params:
            params[feature] = [0, 0, 0]

        w = params[feature]
        elasped = now - w[1]
        upd = scale
        cur_val = w[0]

        w[0] = cur_val + upd
        w[2] += elasped * cur_val + upd
        w[1] = now
