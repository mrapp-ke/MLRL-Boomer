"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for compiling the native library dependency "HDF5" from source.
"""

import platform
from os import environ
from pathlib import Path

from core.build_unit import BuildUnit
from util.io import create_directories
from util.log import Log

from targets.dependencies.native.cmake import Cmake
from targets.dependencies.native.curl import CurlDownload
from targets.dependencies.native.tar import TarExtract
from targets.project import Project
from targets.version_files import VersionTextFile

INSTALL_DIRECTORY = Path('hdf5')


def __get_version() -> str:
    version_file = VersionTextFile(Project.BuildSystem.resource_directory / 'dependencies' / 'hdf5')
    return version_file.version_string.strip()


def __download_and_extract_package(version: str, extract_to: Path) -> Path:
    file_name = f'hdf5-{version}.tar.gz'
    download_url = f'https://github.com/HDFGroup/hdf5/releases/download/hdf5_{version}/{file_name}'
    file = extract_to / file_name
    Log.info(f'Downloading from "{download_url}"...')
    create_directories(extract_to)
    CurlDownload(download_url, file_name=str(file)).run()
    Log.info(f'Extracting file "{file}" into directory "{extract_to}"...')
    TarExtract(file_to_extract=file, into_directory=extract_to).run()
    return extract_to / f'hdf5-{version}'


def __get_macos_arguments() -> list[str]:
    args = []
    deployment_target = environ.get('MACOSX_DEPLOYMENT_TARGET')

    if deployment_target:
        args.append(f'-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}')

    arch_flags = environ.get('ARCHFLAGS', '').split()
    architectures = [arch_flags[i + 1] for i, flag in enumerate(arch_flags[:-1]) if flag == '-arch']

    if architectures:
        args.append(f'-DCMAKE_OSX_ARCHITECTURES={";".join(architectures)}')

    return args


def compile_hdf5(_: BuildUnit):
    """
    Compiles the dependency "HDF5" (see https://github.com/HDFGroup/hdf5) as a shared library and installs it into the
    directory "hdf5" in the project's root directory.
    """
    version = __get_version()
    install_directory = INSTALL_DIRECTORY.resolve()
    source_directory = __download_and_extract_package(version, extract_to=install_directory / 'src')
    Log.info(f'Compiling HDF5 {version} from source directory "{source_directory}"...')
    args = [
        f'-DCMAKE_INSTALL_PREFIX={install_directory}',
        '-DCMAKE_INSTALL_LIBDIR=lib',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP=ON',
        '-Wno-dev',
        '-DBUILD_SHARED_LIBS=ON',
        '-DBUILD_STATIC_LIBS=OFF',
        '-DBUILD_TESTING=OFF',
        '-DHDF5_BUILD_TOOLS=OFF',
        '-DHDF5_BUILD_UTILS=OFF',
        '-DHDF5_BUILD_EXAMPLES=OFF',
        '-DHDF5_BUILD_HL_LIB=OFF',
        '-DHDF5_BUILD_CPP_LIB=OFF',
        '-DHDF5_BUILD_FORTRAN=OFF',
        '-DHDF5_BUILD_JAVA=OFF',
        '-DHDF5_ENABLE_Z_LIB_SUPPORT=OFF',
        '-DHDF5_ENABLE_SZIP_SUPPORT=OFF',
    ]

    if platform.system().lower() == 'darwin':
        args.extend(__get_macos_arguments())

    cmake_build_directory = install_directory / 'build'
    create_directories(cmake_build_directory)
    Cmake('-S', str(source_directory), '-B', str(cmake_build_directory), *args).run()
    Cmake('--build', str(cmake_build_directory), '--config', 'Release', '--parallel').run()
    Cmake('--install', str(cmake_build_directory), '--config', 'Release').run()
