#!/usr/bin/env python
try:
    from .tb_parser import State
except (ValueError, SystemError) as e:
    from tb_parser import State


def is_punct(token, lang):
    if lang == 'en' and token['feat'] in (".", ",", ":", "''", "``"):
        return True
    elif lang == 'ch' and token['feat'] == 'PU':
        return True
    return False


def evaluate(dataset, session, parser, model, ignore_punct=False, lang='en'):
    """

    :param dataset: list
    :param session: tf.Session()
    :param parser: Parser
    :param model: Network
    :param ignore_punct: bool
    :param lang: str
    :return:
    """
    n_uas, n_total = 0, 0
    for data in dataset:
        d = [parser.ROOT] + data
        s = State(d)
        while not s.terminate():
            x = parser.parameterize_x(parser.extract_features(s))
            best, best_action = None, None
            prediction = model.classify(session, x)[0]
            for aid, p in enumerate(prediction):
                action = parser.get_action(aid)
                if s.valid(action) and (best is None or p > best):
                    best, best_action = p, action
            s.transit(best_action)

        for i in range(1, len(d)):
            if ignore_punct and is_punct(d[i], lang):
                continue
            if d[i]['head'] == s.result[i]['head']:
                n_uas += 1
            n_total += 1
    return float(n_uas) / n_total
