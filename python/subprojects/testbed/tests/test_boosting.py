"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from unittest import main

from test_common import CommonIntegrationTests


class BoostingIntegrationTests(CommonIntegrationTests):
    """
    Defines a series of integration tests for the BOOMER algorithm.
    """

    def __init__(self, methodName='runTest'):
        """
        :param methodName: The name of the test method to be executed
        """
        super(BoostingIntegrationTests, self).__init__('boomer', methodName)


if __name__ == '__main__':
    main()
