"""Tests for ot_lib"""
import sys
import os
import numpy as np
from ot_lib import sinkhorn
from
# Needed so that the test file can access src files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

