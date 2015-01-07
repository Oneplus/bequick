#!/usr/bin/env python
import re
import sys
from constituent import Constituent, path

def _extract_predicate(predicate):
    if predicate.terminal():
        return ("PRED=%s" % predicate[0])
    else:
        name = predicate.leaves()
        #print >> sys.stderr, "name=", name, type(name)
        return ("PRED=%s" % "_".join(name))

def _extract_path(predicate, constituent):
    payload = path(constituent, predicate)
    retval = ""
    for d, p in payload:
        retval += p.label()
        if d == "U":
            retval += "|"
        else:
            retval += "!"
    retval = retval[:-1]
    return ("PATH=%s" % retval)

def _extract_phrase_type(constituent):
    return ("PHTP=%s" % constituent.label())

def _extract_position(predicate, constituent):
    if constituent.end < predicate.start:
        return "POSI=Forward"
    else:
        return "POSI=Backward"

def _extract_headword(constituent):
    return ("HEWD=%s" % constituent.head_word)

def _extract_lexicalized_constituent_type(predicate, constituent):
    if predicate.terminal():
        return ("LEXPRE=%s_%s" % (predicate[0], constituent.label()))
    else:
        name = predicate.leaves()
        return ("LEXPRE=%s_%s" % ("_".join(name), constituent.label()))

def _extract_lexicalized_headword(predicate, constituent):
    if predicate.terminal():
        return ("LEXHED=%s_%s" % (predicate[0], constituent.head_word))
    else:
        name = predicate.leaves()
        return ("LEXHED=%s_%s" % ("_".join(name), constituent.head_word))

def extract_feature(predicate, constituent):
    '''
    Parameter
    ---------
    predicate: 
    '''
    return [
            _extract_predicate(predicate),
            _extract_path(predicate, constituent),
            _extract_phrase_type(constituent),
            _extract_position(predicate, constituent),
            _extract_headword(constituent),
            _extract_lexicalized_constituent_type(predicate, constituent),
            _extract_lexicalized_headword(predicate, constituent)
            ]
