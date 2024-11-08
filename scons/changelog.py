"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for validating and updating the project's changelog.
"""
import sys

from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from os.path import isfile
from typing import List, Optional

from versioning import Version, get_current_version

PREFIX_HEADER = '# '

PREFIX_SUB_HEADER = '## '

PREFIX_SUB_SUB_HEADER = '### '

PREFIX_DASH = '- '

PREFIX_ASTERISK = '* '

URL_DOCUMENTATION = 'https://mlrl-boomer.readthedocs.io/en/'

CHANGELOG_FILE_MAIN = '.changelog-main.md'

CHANGELOG_FILE_FEATURE = '.changelog-feature.md'

CHANGELOG_FILE_BUGFIX = '.changelog-bugfix.md'

CHANGELOG_FILE = 'CHANGELOG.md'

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

    def __str__(self) -> str:
        changeset = PREFIX_SUB_SUB_HEADER + self.header + '\n\n'

        for content in self.changes:
            changeset += PREFIX_DASH + content + '\n'

        return changeset


class ReleaseType(Enum):
    """
    Represents the type of a release.
    """
    MAJOR = 'major'
    MINOR = 'feature'
    PATCH = 'bugfix'


@dataclass
class Release:
    """
    A release, consisting of a version, a release date, a type, and several changesets.

    Attributes:
        version:        The version
        release_date:   The release date
        release_type:   The type of the release
        changesets:     A list that stores the changesets
    """
    version: Version
    release_date: date
    release_type: ReleaseType
    changesets: List[Changeset] = field(default_factory=list)

    @staticmethod
    def __format_release_month(month: int) -> str:
        return ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month - 1]

    @staticmethod
    def __format_release_day(day: int) -> str:
        if 11 <= (day % 100) <= 13:
            suffix = 'th'
        else:
            suffix = ['th', 'st', 'nd', 'rd', 'th'][min(day % 10, 4)]

        return str(day) + suffix

    def __format_release_date(self) -> str:
        return self.__format_release_month(self.release_date.month) + '. ' + self.__format_release_day(
            self.release_date.day) + ', ' + str(self.release_date.year)

    def __format_disclaimer(self) -> str:
        if [changeset for changeset in self.changesets if changeset.header.lower() == 'api changes']:
            return ('```{warning}\nThis release comes with API changes. For an updated overview of the available '
                    + 'parameters and command line arguments, please refer to the ' + '[documentation]('
                    + URL_DOCUMENTATION + str(self.version) + ').\n```\n\n')
        return ''

    def __str__(self) -> str:
        release = PREFIX_SUB_HEADER + 'Version ' + str(self.version) + ' (' + self.__format_release_date() + ')\n\n'
        release += 'A ' + self.release_type.value + ' release that comes with the following changes.\n\n'
        release += self.__format_disclaimer()

        for i, changeset in enumerate(self.changesets):
            release += str(changeset) + ('\n' if i < len(self.changesets) else '\n\n')

        return release


def __read_lines(changelog_file: str, skip_if_missing: bool = False) -> List[str]:
    if skip_if_missing and not isfile(changelog_file):
        return []

    with open(changelog_file, mode='r', encoding=CHANGELOG_ENCODING) as file:
        return file.readlines()


def __write_lines(changelog_file: str, lines: List[str]):
    with open(changelog_file, mode='w', encoding=CHANGELOG_ENCODING) as file:
        file.writelines(lines)


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


def __merge_changesets(*changelog_files) -> List[Changeset]:
    changesets_by_header = {}

    for changelog_file in changelog_files:
        for changeset in __parse_changesets(changelog_file):
            merged_changeset = changesets_by_header.setdefault(changeset.header.lower(), changeset)

            if merged_changeset != changeset:
                merged_changeset.changes.extend(changeset.changes)

    return list(changesets_by_header.values())


def __create_release(release_type: ReleaseType, *changelog_files) -> Release:
    return Release(version=get_current_version(),
                   release_date=date.today(),
                   release_type=release_type,
                   changesets=__merge_changesets(*changelog_files))


def __add_release_to_changelog(changelog_file: str, new_release: Release):
    formatted_release = str(new_release)
    print('Adding new release to changelog file "' + changelog_file + '":\n\n' + formatted_release)
    original_lines = __read_lines(changelog_file)
    modified_lines = []
    offset = 0

    for offset, line in enumerate(original_lines):
        if line.startswith(PREFIX_SUB_HEADER):
            break

        modified_lines.append(line)

    modified_lines.append(formatted_release)
    modified_lines.extend(original_lines[offset:])
    __write_lines(changelog_file, modified_lines)


def __clear_changelogs(*changelog_files):
    for changelog_file in changelog_files:
        print('Clearing changelog file "' + changelog_file + '"...')
        __write_lines(changelog_file, [''])


def __update_changelog(release_type: ReleaseType, *changelog_files):
    new_release = __create_release(release_type, *changelog_files)
    __add_release_to_changelog(CHANGELOG_FILE, new_release)
    __clear_changelogs(*changelog_files)


def __get_latest_changelog() -> str:
    changelog = ''
    lines = __read_lines(CHANGELOG_FILE)
    offset = 0

    for offset, line in enumerate(lines):
        if line.startswith(PREFIX_SUB_HEADER):
            break

    for line in lines[offset + 2:]:
        if line.startswith(PREFIX_SUB_HEADER):
            break

        if line.startswith('```{'):
            changelog += '***'
        elif line.startswith('```'):
            changelog = changelog.rstrip('\n')
            changelog += '***\n'
        else:
            changelog += line

    return changelog.rstrip('\n')


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


def update_changelog_main(**_):
    """
    Updates the projects changelog when releasing bugfixes.
    """
    __update_changelog(ReleaseType.MAJOR, CHANGELOG_FILE_MAIN, CHANGELOG_FILE_FEATURE, CHANGELOG_FILE_BUGFIX)


def update_changelog_feature(**_):
    """
    Updates the project's changelog when releasing new features.
    """
    __update_changelog(ReleaseType.MINOR, CHANGELOG_FILE_FEATURE, CHANGELOG_FILE_BUGFIX)


def update_changelog_bugfix(**_):
    """
    Updates the project's changelog when releasing major updates.
    """
    __update_changelog(ReleaseType.PATCH, CHANGELOG_FILE_BUGFIX)


def print_latest_changelog(**_):
    """
    Prints the changelog of the latest release.
    """
    print(__get_latest_changelog())
