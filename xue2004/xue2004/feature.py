#!/usr/bin/env python
import re
import sys
from constituent import Constituent, path

def _meta_predicate(predicate):
    if predicate.terminal():
        return predicate[0]
    else:
        name = predicate.leaves()
        return "_".join(name)

def _meta_predicate_postag(predicate):
    if predicate.terminal():
        return predicate.label()
    else:
        name = [pos for form, pos in predicate.pos()]
        return "_".join(name)

def _meta_phrase_type(constituent):
    return constituent.label()

def _meta_headword(constituent):
    return constituent.head_word

def _meta_voice(predicate):
    words = predicate.root().leaves()
    if predicate.start- 2 >= 0:
        first_id, second_id = predicate.start- 2, predicate.start- 1
        first, second = words[first_id], words[second_id]
        predicate_form = words[predicate.start]
        if first == 'to' and second in ('be', 'get') and not predicate_form.endswith("ing"):
            return "passive"
    if predicate.start- 1 >= 0:
        first_id = predicate.start- 1
        first = words[first_id]
        predicate_form = words[predicate.start]
        if (first in ("am", "is", "was", "are", "been", "being",
                      "get", "got", "gotten", "getting", "gets") and not predicate_form.endswith("ing")):
            return "passive"
    return "active"

def _meta_subcat(predicate):
    parent = predicate.parent()
    retval = "%s->" % parent.label()
    retval += "|".join([kid.label() for kid in parent])
    return retval

def _meta_position(constituent, predicate):
    if constituent.end < predicate.start:
        return "Forward"
    elif constituent.start == predicate.start:
        return "Here"
    else:
        return "Backward"

def _meta_distance(constituent, predicate):
    return len(path(constituent, predicate))

def _extract_path(predicate, constituent):
    payload = path(constituent, predicate)
    retval = ""
    for d, p in payload:
        retval += p.label()
        if d == "U":
            retval += "|"
        else:
            retval += "#"
    retval = retval[:-1]
    return ("PATH=%s" % retval)

def _extract_headword(constituent):
    return ("HEWD=%s" % constituent.head_word)

def _extract_headword_postag(constituent):
    tree = constituent.root()
    words = tree.leaves()
    index = words.index(constituent.head_word)
    postag = tree[tree.leaf_treeposition(index)[:-1]].label()
    return ("HEWDPOS=%s" % postag)

def _extract_predicate(predicate):
    return "PRED=%s" % _meta_predicate(predicate)

def _extract_predicate_postag(predicate):
    return "PRED=%s" % _meta_predicate_postag(predicate)

def _extract_distance(constituent, predicate):
    return "DIST=%d" % _meta_distance(constituent, predicate)

def _extract_phrase_type(constituent):
    return "PHT=%s" % _meta_phrase_type(constituent)

def _extract_position(constituent, predicate):
    return "POS=%s" % _meta_position(constituent, predicate)

def _extract_subcat(predicate):
    return "SUBCAT=%s" % _meta_subcat(predicate)

def _extract_predicate_phrase_type_combo(constituent, predicate):
    return "PRED-PHT=%s-%s" % (_meta_predicate(predicate), _meta_phrase_type(constituent))

def _extract_predicate_headword_combo(constituent, predicate):
    return "PRED-HW=%s-%s" % (_meta_predicate(predicate), _meta_headword(constituent))

def _extract_distance_predicate_phrase(constituent, predicate):
    return "PRED-DIST=%s-%d" % (_meta_predicate(predicate), _meta_distance(constituent, predicate))

def _extract_voice(predicate):
    return "VOICE=%s" % _meta_voice(predicate)

def _extract_voice_position(constituent, predicate):
    return "VOI-POS=%s-%s" % (_meta_voice(predicate), _meta_position(constituent, predicate))

def _extract_lexicalized_constituent_type(constituent, predicate):
    return "LEXPRE=%s-%s" % (_meta_predicate(predicate), constituent.label())

def _extract_lexicalized_headword(constituent, predicate):
    return "LEXHED=%s-%s" % (_meta_predicate(predicate), constituent.head_word)

def _extract_head_of_pp_parent(constituent):
    cat = constituent.parent().label().split("-")[0]
    if cat == "PP":
        return "HEDPP=True"
    else:
        return "HEDPP=False"

def extract_feature(predicate, constituent):
    '''
    Parameter
    ---------
    predicate:
    '''
    return [
            # AI
            _extract_path(predicate, constituent),
            _extract_headword(constituent),
            _extract_headword_postag(constituent),
            _extract_predicate(predicate),
            _extract_predicate_postag(predicate),
            _extract_distance(constituent, predicate),
            _extract_predicate_phrase_type_combo(constituent, predicate),
            _extract_predicate_headword_combo(constituent, predicate),
            _extract_distance_predicate_phrase(constituent, predicate),
            _extract_subcat(predicate),
            # AC
            _extract_phrase_type(constituent),
            _extract_position(predicate, constituent),
            _extract_voice(predicate),
            _extract_voice_position(constituent, predicate),
            _extract_lexicalized_constituent_type(constituent, predicate),
            _extract_lexicalized_headword(constituent, predicate),
            _extract_head_of_pp_parent(constituent)
            ]
