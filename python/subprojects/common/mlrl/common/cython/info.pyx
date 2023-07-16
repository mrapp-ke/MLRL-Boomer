"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


def get_library_version() -> str:
    """
    Returns the version of the library.

    :return: A string that specifies the library version
    """
    cdef string library_version = getLibraryVersion()
    return library_version.decode('UTF-8')
