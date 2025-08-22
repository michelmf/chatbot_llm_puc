"""
Module to define exceptions for the application.
"""


class CorpusDataDoesNotExist(Exception):
    """
    Raise exception when the corpus data is not available.
    """


class IndexingError(Exception):
    """
    Raise exception when there is an error during indexing.
    """


class EnvironmentVariableError(Exception):
    """
    Raise exception when an environment variable is not set.
    """
