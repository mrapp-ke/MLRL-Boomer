"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for compiling native library dependencies on macOS.
"""
import platform

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


def __get_download_url_and_file_name(github_api: GithubApi) -> Tuple[str, str]:
    assets = github_api.open_repository('llvm/llvm-project').get_latest_release().get_assets()
    package_name = 'openmp'
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


def compile_libomp(build_unit: BuildUnit):
    """
    Compiles the dependency "libomp" (see https://github.com/Homebrew/homebrew-core/tree/master/Formula/lib/libomp.rb).
    """
    if platform.system().lower() != 'darwin':
        Log.info('Target may only be run on macOS!')  # TODO Error

    Log.info('Determining download URL of the latest release...')
    github_api = GithubApi(build_unit).set_token_from_env()
    download_url, file_name = __get_download_url_and_file_name(github_api)
    Log.info('Downloading from "%s"...', download_url)
    authorization_header = 'token ' + github_api.token if github_api.token else None
    CurlDownload(download_url, authorization_header=authorization_header, file_name=file_name).run()

    file_name_without_suffix = file_name[:len(SUFFIX_SRC_TAR_XZ)]
    extract_directory = file_name_without_suffix
    Log.info('Unpacking file "%s" into directory "%s"...', file_name, extract_directory)
    create_directories(extract_directory)
    TarExtract(file_to_extract=file_name, into_directory=extract_directory).run()

    source_directory = path.join(extract_directory, file_name_without_suffix + SUFFIX_SRC)
    Log.info('Compiling from source directory "%s"...', source_directory)
    file_name_parts = file_name_without_suffix.split('-')
    package_name = file_name_parts[0]
    package_version = file_name_parts[-1]
    args = [
        '-DCMAKE_INSTALL_PREFIX=' + path.join('opt', package_name, package_version),
        '-DCMAKE_INSTALL_LIBDIR=lib',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_FIND_FRAMEWORK=LAST',
        #'-DCMAKE_VERBOSE_MAKEFILE=ON',
        #'-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=/opt/homebrew/Library/Homebrew/cmake/trap_fetchcontent_provider.cmake',
        #'-Wno-dev',
        #'-DBUILD_TESTING=OFF',
        #'-DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX14.sdk',
        '-DLIBOMP_INSTALL_ALIASES=OFF',
    ]
    build_directory = path.join('build', 'shared')
    Cmake('-S', source_directory, '-B', build_directory, *args).run()
    Cmake('--build', build_directory).run()
    Cmake('--install', build_directory).run()
