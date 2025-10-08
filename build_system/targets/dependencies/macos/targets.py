"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for compiling native library dependencies on macOS.
"""
import platform
import shutil

from pathlib import Path
from typing import Any, Optional, Tuple

from core.build_unit import BuildUnit
from util.io import create_directories
from util.log import Log
from util.pygithub import GithubApi

from targets.dependencies.macos.cmake import Cmake
from targets.dependencies.macos.curl import CurlDownload
from targets.dependencies.macos.tar import TarExtract

SUFFIX_SRC = '.src'

SUFFIX_SRC_TAR_XZ = SUFFIX_SRC + '.tar.xz'


def __get_download_url_and_file_name_from_release(release: Any, package_name: str) -> Optional[Any]:
    # Ignore release candidates
    if not '-rc' in release.title:
        assets = release.get_assets()
        prefix = package_name + '-'

        for asset in assets:
            asset_name = asset.name

            if asset_name.startswith(prefix) and asset_name.endswith(SUFFIX_SRC_TAR_XZ):
                return asset

    return None


def __get_download_url_and_file_name(github_api: GithubApi, package_name: str) -> Tuple[Optional[str], Optional[str]]:
    repository = github_api.open_repository('llvm/llvm-project')
    asset = __get_download_url_and_file_name_from_release(repository.get_latest_release(), package_name)

    if not asset:
        for release in repository.get_all_releases():
            asset = __get_download_url_and_file_name_from_release(release, package_name)

            if asset:
                break

    if not asset:
        Log.error('Failed to identify asset to be downloaded!')
        return None, None

    return asset.browser_download_url, asset.name


def __download_package(github_api: GithubApi, package_name: str) -> Optional[str]:
    Log.info('Determining download URL of the latest release of package "%s"...', package_name)
    download_url, file_name = __get_download_url_and_file_name(github_api, package_name=package_name)

    if download_url and file_name:
        Log.info('Downloading from "%s"...', download_url)
        authorization_header = 'token ' + github_api.token if github_api.token else None
        CurlDownload(download_url, authorization_header=authorization_header, file_name=file_name).run()
        return file_name

    return None


def __download_and_extract_package(github_api: GithubApi, package_name: str, extract_to: Path) -> Optional[Path]:
    file_name = __download_package(github_api, package_name=package_name)

    if file_name:
        file_name_without_suffix = file_name[:-len(SUFFIX_SRC_TAR_XZ)]
        extract_directory = extract_to / file_name_without_suffix
        Log.info('Extracting file "%s" into directory "%s"...', file_name, extract_directory)
        create_directories(extract_directory)
        TarExtract(file_to_extract=Path(file_name), into_directory=extract_directory).run()
        return extract_directory / (file_name_without_suffix + SUFFIX_SRC)

    return None


def compile_libomp(build_unit: BuildUnit):
    """
    Compiles the dependency "libomp" (see https://github.com/Homebrew/homebrew-core/tree/master/Formula/lib/libomp.rb).
    """
    if platform.system().lower() != 'darwin':
        Log.error('Target may only be run on macOS!')

    github_api = GithubApi(build_unit).set_token_from_env()
    build_directory = Path('libomp')
    cmake_directory = __download_and_extract_package(github_api, package_name='cmake', extract_to=build_directory)
    openmp_directory = __download_and_extract_package(github_api, package_name='openmp', extract_to=build_directory)

    if cmake_directory and openmp_directory:
        shutil.copytree(cmake_directory / 'Modules', openmp_directory / 'cmake', dirs_exist_ok=True)
        Log.info('Compiling from source directory "%s"...', openmp_directory)
        args = [
            '-DCMAKE_INSTALL_PREFIX=' + str(build_directory),
            '-DCMAKE_INSTALL_LIBDIR=lib',
            '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_FIND_FRAMEWORK=LAST',
            '-DCMAKE_VERBOSE_MAKEFILE=ON',
            '-Wno-dev',
            '-DBUILD_TESTING=OFF',
            '-DLIBOMP_INSTALL_ALIASES=OFF',
        ]
        cmake_build_directory = build_directory / 'build' / 'shared'
        create_directories(cmake_build_directory)
        Cmake('-S', str(openmp_directory), '-B', str(cmake_build_directory), *args).run()
        Cmake('--build', str(cmake_build_directory)).run()
        Cmake('--install', str(cmake_build_directory)).run()
