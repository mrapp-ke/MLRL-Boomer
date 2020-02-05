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


def optional_string(s):
    if s is None or s.lower() == 'none':
        return None
    return s
