"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for triggering readthedocs builds.
"""
from core.build_unit import BuildUnit
from util.log import Log

from targets.documentation.rdt.rdt import ReadTheDocsApi
from targets.project import Project


def trigger_readthedocs_build(build_unit: BuildUnit):
    """
    Triggers a readthedocs build.
    """
    project_version = Project.version()
    rdt_version = 'latest' if project_version.dev else project_version
    Log.info('Triggering readthedocs build for version "%s"...', rdt_version)
    ReadTheDocsApi(build_unit).set_project('mlrl-boomer').trigger_build(rdt_version)
