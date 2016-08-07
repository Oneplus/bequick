#!/usr/bin/env python
__boc__ = "__boc__"
__eoc__ = "__eoc__"
__bot__ = "__bot__"
__eot__ = "__eot__"
__bow__ = "__bow__"


def extract_join_features(chars, charts, i, state):
    """
    Extract features for JOIN action

    Parameters
    ----------
    chars : list(str)
        The list of characters
    charts : list(str)
        The list of character types
    i : int
        The index
    state : State
        The state

    Returns
    -------
    ret : list(str)
        The list of feature strings
    """
    L = len(chars)
    prev_ch = chars[i - 1] if i - 1 >= 0 else __boc__
    curr_ch = chars[i]
    next_ch = chars[i + 1] if i + 1 < L else __eoc__
    prev_cht = charts[i - 1] if i - 1 >= 0 else __bot__
    curr_cht = charts[i]
    next_cht = charts[i + 1] if i + 1 < L else __eot__

    ret = ["1=c[-1]=%s" % prev_ch,
           "2=c[0]=%s" % curr_ch,
           "3=c[1]=%s" % next_ch,
           "4=ct[-1]=%s" % prev_cht,
           "5=ct[0]=%s" % curr_cht,
           "6=ct[1]=%s" % next_cht,
           "7=c[-1]c[0]=%s%s" % (prev_ch, curr_ch),
           "8=c[0]c[1]=%s%s" % (curr_ch, next_ch),
           "9=ct[-1]ct[0]=%s%s" % (prev_cht, curr_cht),
           "10=ct[0]ct[1]=%s%s" % (curr_cht, next_cht),
           "11=c[-1]c[0]c[1]=%s%s%s" % (prev_ch, curr_ch, next_ch),
           "12=ct[-1]ct[0]ct[1]=%s%s%s" % (prev_cht, curr_cht, next_cht)]
    return ret


def extract_separate_features(chars, charts, i, state):
    """
    Extract features for the SEPARATE actions

    Parameters
    ----------
    chars : list(str)
        The list of characters
    charts : list(str)
        The list of character types
    i : int
        The index
    state : State
        The source state

    Returns
    -------
    ret : list(str)
        The list of feature string
    """
    L = len(chars)

    prev_ch = chars[i - 1] if i - 1 >= 0 else __boc__
    curr_ch = chars[i]
    next_ch = chars[i + 1] if i + 1 < L else __eoc__
    prev_cht = charts[i - 1] if i - 1 >= 0 else __bot__
    curr_cht = charts[i]
    next_cht = charts[i + 1] if i + 1 < L else __eot__
    curr_w = "".join(chars[state.curr.index: i + 1])
    curr_w_len = i + 1 - state.curr.index

    ret = ["1=c[-1]=%s" % prev_ch,
           "2=c[0]=%s" % curr_ch,
           "3=c[1]=%s" % next_ch,
           "4=ct[-1]=%s" % prev_cht,
           "5=ct[0]=%s" % curr_cht,
           "6=ct[1]=%s" % next_cht,
           "7=c[-1]c[0]=%s%s" % (prev_ch, curr_ch),
           "8=c[0]c[1]=%s%s" % (curr_ch, next_ch),
           "9=ct[-1]ct[0]=%s%s" % (prev_cht, curr_cht),
           "10=ct[0]ct[1]=%s%s" % (curr_cht, next_cht),
           "11=c[-1]c[0]c[1]=%s%s%s" % (prev_ch, curr_ch, next_ch),
           "12=ct[-1]ct[0]ct[1]=%s%s%s" % (prev_cht, curr_cht, next_cht),
           "13=w[0]=%s" % curr_w]

    if curr_w_len == 1:
        ret.append("14=single-char")
    else:
        ret.append("15=first=%s=last=%s" % (chars[state.curr.index], chars[i]))
        ret.append("16=first=%s=len[0]=%d" % (chars[state.curr.index], curr_w_len))
        ret.append("17=last=%s=len[0]=%d" % (chars[i], curr_w_len))

    if state.prev is not None:
        prev_w = "".join(chars[state.prev.curr.index: state.curr.index])
        prev_w_len = state.curr.index - state.prev.curr.index
        ret.append("18=word[-1]=%s-word[0]=%s" % (prev_w, curr_w))
        ret.append("19=prevch=%s-word[0]=%s" % (chars[state.curr.index - 1], curr_w))
        ret.append("20=word[-1]=%s-prevch=%s" % (prev_w, chars[state.curr.index - 1]))
        ret.append("21=word[-1]=%s-len[0]=%d" % (prev_w, curr_w_len))
        ret.append("22=word[0]=%s-len[-1]=%d" % (curr_w, prev_w_len))
    return ret


def extract_features(action, chars, charts, i, state):
    if action == 's':
        return extract_separate_features(chars, charts, i, state)
    else:
        return extract_join_features(chars, charts, i, state)
