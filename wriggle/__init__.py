# -*- coding: utf-8 -*-

"""Top-level package for Wriggle."""

__author__ = """J. Michael Burgess"""
__email__ = 'jburgess@mpe.mpg.de'

from .data_prep import GRBInterval, GRBDatum
from .stan_models.stan_model import get_stan_model

from . import _version
__version__ = _version.get_versions()['version']
