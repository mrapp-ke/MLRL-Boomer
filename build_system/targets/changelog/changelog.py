"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides actions for validating and updating the project's changelog.
"""
from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from functools import cached_property
from typing import List, Optional

from core.build_unit import BuildUnit
from util.io import TextFile
from util.log import Log

from targets.project import Project
from targets.version_files import Version

CHANGESET_FILE_MAIN = '.changelog-main.md'

CHANGESET_FILE_FEATURE = '.changelog-feature.md'

CHANGESET_FILE_BUGFIX = '.changelog-bugfix.md'


class LineType(Enum):
    """
    Represents different types of lines that may occur in a changeset.
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
        if line.startswith(Line.PREFIX_HEADER):
            return LineType.HEADER
        if line.startswith(Line.PREFIX_DASH) or line.startswith(Line.PREFIX_ASTERISK):
            return LineType.ENUMERATION
        return None


@dataclass
class Line:
    """
    A single line in a changeset.

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

    PREFIX_HEADER = '# '

    PREFIX_DASH = '- '

    PREFIX_ASTERISK = '* '

    @staticmethod
    def parse(line: str, line_number: int) -> 'Line':
        """
        Parses and returns a single line in a changeset.

        :param line:        The line to be parsed
        :param line_number: The number of the line to parsed (starting at 1)
        :return:            The `Line` that has been created
        """
        line = line.strip('\n')
        line_type = LineType.parse(line)

        if not line_type:
            raise ValueError('Line ' + str(line_number)
                             + ' is invalid: Must be blank, a top-level header (starting with "' + Line.PREFIX_HEADER
                             + '"), or an enumeration (starting with "' + Line.PREFIX_DASH + '" or "'
                             + Line.PREFIX_ASTERISK + '"), but is "' + line + '"')

        content = line

        if line_type != LineType.BLANK:
            content = line.lstrip(Line.PREFIX_HEADER).lstrip(Line.PREFIX_DASH).lstrip(Line.PREFIX_ASTERISK)

            if not content or content.isspace():
                raise ValueError('Line ' + str(line_number) + ' is is invalid: Content must not be blank, but is "'
                                 + line + '"')

        return Line(line_number=line_number, line_type=line_type, line=line, content=content)


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
        changeset = '### ' + self.header + '\n\n'

        for content in self.changes:
            changeset += Line.PREFIX_DASH + content + '\n'

        return changeset


class ChangesetFile(TextFile):
    """
    A file that stores several changesets.
    """

    def __validate_line(self, current_line: Optional[Line], previous_line: Optional[Line]):
        current_line_is_enumeration = current_line and current_line.line_type == LineType.ENUMERATION

        if current_line_is_enumeration and not previous_line:
            raise ValueError('File "' + self.file + '" must start with a top-level header (starting with "'
                             + Line.PREFIX_HEADER + '")')

        current_line_is_header = current_line and current_line.line_type == LineType.HEADER
        previous_line_is_header = previous_line and previous_line.line_type == LineType.HEADER

        if (current_line_is_header and previous_line_is_header) or (not current_line and previous_line_is_header):
            raise ValueError('Header "' + previous_line.line + '" at line ' + str(previous_line.line_number)
                             + ' of file "' + self.file + '" is not followed by any content')

    @cached_property
    def parsed_lines(self) -> List[Line]:
        """
        The lines in the changelog as `Line` objects.
        """
        parsed_lines = []

        for i, line in enumerate(self.lines):
            current_line = Line.parse(line, line_number=i + 1)

            if current_line.line_type != LineType.BLANK:
                parsed_lines.append(current_line)

        return parsed_lines

    @cached_property
    def changesets(self) -> List[Changeset]:
        """
        A list that contains all changesets in the changelog.
        """
        changesets = []

        for line in self.parsed_lines:
            if line.line_type == LineType.HEADER:
                changesets.append(Changeset(header=line.content))
            elif line.line_type == LineType.ENUMERATION:
                current_changeset = changesets[-1]
                current_changeset.changes.append(line.content)

        return changesets

    def validate(self):
        """
        Validates the changelog.
        """
        previous_line = None

        for current_line in self.parsed_lines:
            if current_line.line_type != LineType.BLANK:
                self.__validate_line(current_line=current_line, previous_line=previous_line)
                previous_line = current_line

        self.__validate_line(current_line=None, previous_line=previous_line)

    def write_lines(self, *lines: str):
        super().write_lines(*lines)

        try:
            del self.parsed_lines
        except AttributeError:
            pass

        try:
            del self.changesets
        except AttributeError:
            pass


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

    URL_DOCUMENTATION = 'https://mlrl-boomer.readthedocs.io/en/'

    PREFIX_SUB_HEADER = '## '

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
                    + self.URL_DOCUMENTATION + str(self.version) + ').\n```\n\n')
        return ''

    def __str__(self) -> str:
        release = self.PREFIX_SUB_HEADER + 'Version ' + str(
            self.version) + ' (' + self.__format_release_date() + ')\n\n'
        release += 'A ' + self.release_type.value + ' release that comes with the following changes.\n\n'
        release += self.__format_disclaimer()

        for i, changeset in enumerate(self.changesets):
            release += str(changeset) + ('\n' if i < len(self.changesets) else '\n\n')

        return release


class ChangelogFile(TextFile):
    """
    The file that stores the project's changelog.
    """

    def __init__(self):
        super().__init__('CHANGELOG.md')

    def add_release(self, release: Release):
        """
        Adds a new release to the project's changelog.

        :param release: The release to be added
        """
        formatted_release = str(release)
        Log.info('Adding new release to changelog file "%s":\n\n%s', self.file, formatted_release)
        original_lines = self.lines
        modified_lines = []
        offset = 0

        for offset, line in enumerate(original_lines):
            if line.startswith(Release.PREFIX_SUB_HEADER):
                break

            modified_lines.append(line)

        modified_lines.append(formatted_release)
        modified_lines.extend(original_lines[offset:])
        self.write_lines(*modified_lines)

    @property
    def latest(self) -> str:
        """
        The latest release in the changelog.
        """
        release = ''
        lines = self.lines
        offset = 0

        for offset, line in enumerate(lines):
            if line.startswith(Release.PREFIX_SUB_HEADER):
                break

        for line in lines[offset + 2:]:
            if line.startswith(Release.PREFIX_SUB_HEADER):
                break

            if line.startswith('```{'):
                release += '***'
            elif line.startswith('```'):
                release = release.rstrip('\n')
                release += '***\n'
            else:
                release += line

        return release.rstrip('\n')


def __validate_changeset(changeset_file: str):
    try:
        Log.info('Validating changeset file "%s"...', changeset_file)
        ChangesetFile(changeset_file, accept_missing=True).validate()
    except ValueError as error:
        Log.error('Changeset file "%s" is malformed!\n\n%s', changeset_file, str(error))


def __merge_changesets(*changeset_files) -> List[Changeset]:
    changesets_by_header = {}

    for changeset_file in changeset_files:
        for changeset in ChangesetFile(changeset_file).changesets:
            merged_changeset = changesets_by_header.setdefault(changeset.header.lower(), changeset)

            if merged_changeset != changeset:
                merged_changeset.changes.extend(changeset.changes)

    return list(changesets_by_header.values())


def __update_changelog(release_type: ReleaseType, *changeset_files):
    merged_changesets = __merge_changesets(*changeset_files)
    new_release = Release(version=Project.version(release=True),
                          release_date=date.today(),
                          release_type=release_type,
                          changesets=merged_changesets)
    ChangelogFile().add_release(new_release)

    for changeset_file in changeset_files:
        ChangesetFile(changeset_file).clear()


def validate_changelog_bugfix(_: BuildUnit):
    """
    Validates the changelog file that lists bugfixes.
    """
    __validate_changeset(CHANGESET_FILE_BUGFIX)


def validate_changelog_feature(_: BuildUnit):
    """
    Validates the changelog file that lists new features.
    """
    __validate_changeset(CHANGESET_FILE_FEATURE)


def validate_changelog_main(_: BuildUnit):
    """
    Validates the changelog file that lists major updates.
    """
    __validate_changeset(CHANGESET_FILE_MAIN)


def update_changelog_main(_: BuildUnit):
    """
    Updates the projects changelog when releasing bugfixes.
    """
    __update_changelog(ReleaseType.MAJOR, CHANGESET_FILE_MAIN, CHANGESET_FILE_FEATURE, CHANGESET_FILE_BUGFIX)


def update_changelog_feature(_: BuildUnit):
    """
    Updates the project's changelog when releasing new features.
    """
    __update_changelog(ReleaseType.MINOR, CHANGESET_FILE_FEATURE, CHANGESET_FILE_BUGFIX)


def update_changelog_bugfix(_: BuildUnit):
    """
    Updates the project's changelog when releasing major updates.
    """
    __update_changelog(ReleaseType.PATCH, CHANGESET_FILE_BUGFIX)


def print_current_version(_: BuildUnit):
    """
    Prints the project's current version.
    """
    return Log.info('%s', str(Project.version(release=True)))


def print_latest_changelog(_: BuildUnit):
    """
    Prints the changelog of the latest release.
    """
    Log.info('%s', ChangelogFile().latest)
