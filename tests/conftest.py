"""Disable the on-disk signal cache during tests.

The cache keys on (qid, contexts, config signature) but not on the fact source, so two
tests that use different fact sources on the same example could otherwise collide. Real
runs use a single fact source per run, so this only matters for the test suite.
"""
import os

os.environ["FRANQ_SIGNAL_CACHE"] = "0"
