"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for validating and updating the project's changelog.
"""
import sys

from dataclasses import dataclass, field
from enum import Enum, auto
from os.path import isfile
from typing import List, Optional

PREFIX_HEADER = '# '

PREFIX_DASH = '- '

PREFIX_ASTERISK = '* '

CHANGELOG_FILE_MAIN = '.changelog-main.md'

CHANGELOG_FILE_FEATURE = '.changelog-feature.md'

CHANGELOG_FILE_BUGFIX = '.changelog-bugfix.md'

CHANGELOG_ENCODING = 'utf-8'


class LineType(Enum):
    """
    Represents different types of lines that may occur in a changelog.
    """
    BLANK = auto()
    HEADER = auto()
    ENUMERATION = auto()

    @staticmethod
    def parse(line: str) -> Optional['LineType']:
        """
        Parses a given line and returns its type.

        :return: The type of the given line or None, if the line is invalid
        """
        if not line or line.isspace():
            return LineType.BLANK
        if line.startswith(PREFIX_HEADER):
            return LineType.HEADER
        if line.startswith(PREFIX_DASH) or line.startswith(PREFIX_ASTERISK):
            return LineType.ENUMERATION
        return None


@dataclass
class Changeset:
    """
    A changeset, consisting of a header and textual descriptions of several changes.

    Attributes:
        header:     The header of the changeset
        changes:    A list that stores the textual descriptions of the changes
    """
    header: str
    changes: List[str] = field(default_factory=list)


@dataclass
class Line:
    """
    A single line in a changelog.

    Attributes:
        line_number:    The line number, starting at 1
        line_type:      The type of the line
        line:           The original content of the line
        content:        The content of the line with Markdown keywords being stripped away
    """
    line_number: int
    line_type: LineType
    line: str
    content: str


def __read_lines(changelog_file: str, skip_if_missing: bool = False) -> List[str]:
    if skip_if_missing and not isfile(changelog_file):
        return []

    with open(changelog_file, mode='r', encoding=CHANGELOG_ENCODING) as file:
        return file.readlines()


def __parse_line(changelog_file: str, line_number: int, line: str) -> Line:
    line = line.strip('\n')
    line_type = LineType.parse(line)

    if not line_type:
        print('Line ' + str(line_number) + ' of file "' + changelog_file
              + '" is invalid: Must be blank, a top-level header (starting with "' + PREFIX_HEADER
              + '"), or an enumeration (starting with "' + PREFIX_DASH + '" or "' + PREFIX_ASTERISK + '"), but is "'
              + line + '"')
        sys.exit(-1)

    content = line

    if line_type != LineType.BLANK:
        content = line.lstrip(PREFIX_HEADER).lstrip(PREFIX_DASH).lstrip(PREFIX_ASTERISK)

        if not content or content.isspace():
            print('Line ' + str(line_number) + ' of file "' + changelog_file
                  + '" is is invalid: Content must not be blank, but is "' + line + '"')
            sys.exit(-1)

    return Line(line_number=line_number, line_type=line_type, line=line, content=content)


def __validate_line(changelog_file: str, current_line: Optional[Line], previous_line: Optional[Line]):
    current_line_is_enumeration = current_line and current_line.line_type == LineType.ENUMERATION

    if current_line_is_enumeration and not previous_line:
        print('File "' + changelog_file + '" must start with a top-level header (starting with "' + PREFIX_HEADER
              + '")')
        sys.exit(-1)

    current_line_is_header = current_line and current_line.line_type == LineType.HEADER
    previous_line_is_header = previous_line and previous_line.line_type == LineType.HEADER

    if (current_line_is_header and previous_line_is_header) or (not current_line and previous_line_is_header):
        print('Header "' + previous_line.line + '" at line ' + str(previous_line.line_number) + ' of file "'
              + changelog_file + '" is not followed by any content')
        sys.exit(-1)


def __parse_lines(changelog_file: str, lines: List[str]) -> List[Line]:
    previous_line = None
    parsed_lines = []

    for i, line in enumerate(lines):
        current_line = __parse_line(changelog_file=changelog_file, line_number=(i + 1), line=line)

        if current_line.line_type != LineType.BLANK:
            __validate_line(changelog_file=changelog_file, current_line=current_line, previous_line=previous_line)
            previous_line = current_line
            parsed_lines.append(current_line)

    __validate_line(changelog_file=changelog_file, current_line=None, previous_line=previous_line)
    return parsed_lines


def __parse_changesets(changelog_file: str, skip_if_missing: bool = False) -> List[Changeset]:
    changesets = []
    lines = __parse_lines(changelog_file, __read_lines(changelog_file, skip_if_missing=skip_if_missing))

    for line in lines:
        if line.line_type == LineType.HEADER:
            changesets.append(Changeset(header=line.content))
        elif line.line_type == LineType.ENUMERATION:
            current_changeset = changesets[-1]
            current_changeset.changes.append(line.content)

    return changesets


def __validate_changelog(changelog_file: str):
    print('Validating changelog file "' + changelog_file + '"...')
    __parse_changesets(changelog_file, skip_if_missing=True)


def validate_changelog_bugfix(**_):
    """
    Validates the changelog file that lists bugfixes.
    """
    __validate_changelog(CHANGELOG_FILE_BUGFIX)


def validate_changelog_feature(**_):
    """
    Validates the changelog file that lists new features.
    """
    __validate_changelog(CHANGELOG_FILE_FEATURE)


def validate_changelog_main(**_):
    """
    Validates the changelog file that lists major updates.
    """
    __validate_changelog(CHANGELOG_FILE_MAIN)
