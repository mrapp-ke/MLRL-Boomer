#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides functions for parsing command line arguments.
"""
import logging as log


def log_level(s):
    s = s.lower()
    if s == 'debug':
        return log.DEBUG
    elif s == 'info':
        return log.INFO
    elif s == 'warn' or s == 'warning':
        return log.WARN
    elif s == 'error':
        return log.ERROR
    elif s == 'critical' or s == 'fatal':
        return log.CRITICAL
    elif s == 'notset':
        return log.NOTSET
    raise ValueError('Invalid argument given for parameter \'--log-level\': ' + str(s))


def boolean_string(s):
    s = s.lower()

    if s == 'false':
        return False
    if s == 'true':
        return True
    raise ValueError('Invalid boolean argument given: ' + str(s))


def optional_string(s):
    if s is None or s.lower() == 'none':
        return None
    return s


def string_list(s):
    return [x.strip() for x in s.split(',')]


def int_list(s):
    return [int(x) for x in string_list(s)]


def float_list(s):
    return [float(x) for x in string_list(s)]
