"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for compiling native library dependencies on macOS.
"""
import platform
import shutil

from os import path
from typing import Tuple

from core.build_unit import BuildUnit
from util.io import create_directories
from util.log import Log

from targets.dependencies.macos.cmake import Cmake
from targets.dependencies.macos.curl import CurlDownload
from targets.dependencies.macos.tar import TarExtract
from targets.dependencies.pygithub import GithubApi

SUFFIX_SRC = '.src'

SUFFIX_SRC_TAR_XZ = SUFFIX_SRC + '.tar.xz'


def __get_download_url_and_file_name(github_api: GithubApi, package_name: str) -> Tuple[str, str]:
    assets = github_api.open_repository('llvm/llvm-project').get_latest_release().get_assets()
    prefix = package_name + '-'
    download_asset = None

    for asset in assets:
        asset_name = asset.name

        if asset_name.startswith(prefix) and asset_name.endswith(SUFFIX_SRC_TAR_XZ):
            download_asset = asset
            break

    if not download_asset:
        Log.error('Failed to identify asset to be downloaded!')

    return download_asset.browser_download_url, download_asset.name


def __download_package(github_api: GithubApi, package_name: str) -> str:
    Log.info('Determining download URL of the latest release of package "%s"...', package_name)
    download_url, file_name = __get_download_url_and_file_name(github_api, package_name=package_name)
    Log.info('Downloading from "%s"...', download_url)
    authorization_header = 'token ' + github_api.token if github_api.token else None
    CurlDownload(download_url, authorization_header=authorization_header, file_name=file_name).run()
    return file_name


def __download_and_extract_package(github_api: GithubApi, package_name: str, extract_to: str) -> str:
    file_name = __download_package(github_api, package_name=package_name)
    file_name_without_suffix = file_name[:-len(SUFFIX_SRC_TAR_XZ)]
    extract_directory = path.join(extract_to, file_name_without_suffix)
    Log.info('Extracting file "%s" into directory "%s"...', file_name, extract_directory)
    create_directories(extract_directory)
    TarExtract(file_to_extract=file_name, into_directory=extract_directory).run()
    return path.join(extract_directory, file_name_without_suffix + SUFFIX_SRC)


def compile_libomp(build_unit: BuildUnit):
    """
    Compiles the dependency "libomp" (see https://github.com/Homebrew/homebrew-core/tree/master/Formula/lib/libomp.rb).
    """
    if platform.system().lower() != 'darwin':
        Log.error('Target may only be run on macOS!')

    github_api = GithubApi(build_unit).set_token_from_env()
    build_directory = 'libomp'
    cmake_directory = __download_and_extract_package(github_api, package_name='cmake', extract_to=build_directory)
    openmp_directory = __download_and_extract_package(github_api, package_name='openmp', extract_to=build_directory)
    shutil.copytree(path.join(cmake_directory, 'Modules'), path.join(openmp_directory, 'cmake'), dirs_exist_ok=True)

    Log.info('Compiling from source directory "%s"...', openmp_directory)
    args = [
        '-DCMAKE_INSTALL_PREFIX=' + build_directory,
        '-DCMAKE_INSTALL_LIBDIR=lib',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_FIND_FRAMEWORK=LAST',
        '-DCMAKE_VERBOSE_MAKEFILE=ON',
        '-Wno-dev',
        '-DBUILD_TESTING=OFF',
        '-DLIBOMP_INSTALL_ALIASES=OFF',
    ]
    cmake_build_directory = path.join(build_directory, 'build', 'shared')
    create_directories(cmake_build_directory)
    Cmake('-S', openmp_directory, '-B', cmake_build_directory, *args).run()
    Cmake('--build', cmake_build_directory).run()
    Cmake('--install', cmake_build_directory).run()
