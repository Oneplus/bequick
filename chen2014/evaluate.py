#!/usr/bin/env python
from __future__ import division
try:
    from .tb_parser import State
    from .instance_builder import InstanceBuilder as IB
except (ValueError, SystemError) as e:
    from tb_parser import State
    from instance_builder import InstanceBuilder as IB


def is_punct(lang, extra):
    if extra is None:
        return False
    if lang == 'en' and extra in (".", ",", ":", "''", "``"):
        return True
    elif lang == 'ch' and extra == 'PU':
        return True
    return False


def evaluate(dataset, session, parser, model, extras=None, ignore_punct=False, lang='en'):
    """

    :param dataset: list
    :param session: tf.Session()
    :param parser: Parser
    :param model: Network
    :param extras: list or None
    :param ignore_punct: bool
    :param lang: str
    :return:
    """
    n_uas, n_las, n_total = 0, 0, 0
    for n, data in enumerate(dataset):
        s = State(data)
        while not s.terminate():
            x = parser.parameterize_x(s)
            prediction = model.classify(session, x)[0]
            best, best_action = None, None
            for aid, p in enumerate(prediction):
                if parser.system.valid(s, aid) and (best is None or p > best):
                    best, best_action = p, aid
            parser.system.transit(s, best_action)

        for i in range(1, len(data)):
            if extras is not None and ignore_punct and is_punct(lang, extras[n][i]):
                continue
            if data[i][IB.HED] == s.result[i][State.HED]:
                n_uas += 1
                if data[i][IB.DEPREL] == s.result[i][State.DEPREL]:
                    n_las += 1
            n_total += 1
    return n_uas / n_total, n_las / n_total
