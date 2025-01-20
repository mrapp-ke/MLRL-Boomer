"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "curl".
"""
from typing import Optional

from util.cmd import Command
from util.run import Program


class CurlDownload(Program):
    """
    Allows to run the external program "curl" for downloading a file from a specific URL.
    """

    def __init__(self,
                 url: str,
                 authorization_header: Optional[str] = None,
                 file_name: Optional[str] = None,
                 follow_redirects: bool = True):
        """
        :param url:                     The URL of the file to be downloaded
        :param authorization_header:    The authorization header to be set or None, if no such header should be set
        :param file_name:               The name of the file to be saved or None, if the original name should be used
        :param follow_redirects:        True, if redirects should be followed, False otherwise
        """
        super().__init__('curl')
        self.add_conditional_arguments(follow_redirects, '--location')
        self.add_conditional_arguments(file_name is not None, '-o', file_name)
        self.add_conditional_arguments(file_name is None, '-O')
        self.add_arguments(url)
        self.use_authorization = authorization_header is not None
        self.add_conditional_arguments(self.use_authorization, '-H', 'Authorization: ' + str(authorization_header))
        self.install_program(False)
        self.print_arguments(not self.use_authorization)

    def __str__(self) -> str:
        print_options = Command.PrintOptions()
        print_options.print_arguments = not self.use_authorization
        return print_options.format(self)
