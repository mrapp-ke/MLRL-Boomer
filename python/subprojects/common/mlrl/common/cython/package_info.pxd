from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from mlrl.common.cython._types cimport uint32


cdef extern from "mlrl/common/library_info.hpp" nogil:

    cdef cppclass BuildOption"ILibraryInfo::BuildOption":

        string option

        string description

        string value


    cdef cppclass HardwareResource"ILibraryInfo::HardwareResource":

        string resource

        string info


ctypedef void (*BuildOptionVisitor)(const BuildOption&)

ctypedef void (*HardwareResourceVisitor)(const HardwareResource&)


cdef extern from "mlrl/common/library_info.hpp" nogil:

    cdef cppclass ILibraryInfo:

        string getLibraryName() const

        string getLibraryVersion() const

        string getTargetArchitecture() const

        void visitBuildOptions(BuildOptionVisitor visitor) const

        void visitHardwareResources(HardwareResourceVisitor visitor) const


    unique_ptr[ILibraryInfo] getLibraryInfo()

    bool isMultiThreadingSupportEnabled()

    uint32 getNumCpuCores()

    bool isGpuSupportEnabled()

    bool isGpuAvailable()

    vector[string] getGpuDevices()


cdef extern from *:
    """
    #include "mlrl/common/library_info.hpp"


    typedef void (*BuildOptionCythonVisitor)(void*, const ILibraryInfo::BuildOption&);

    typedef void (*HardwareResourceCythonVisitor)(void*, const ILibraryInfo::HardwareResource&);

    static inline ILibraryInfo::BuildOptionVisitor wrapBuildOptionVisitor(void* self,
                                                                          BuildOptionCythonVisitor visitor) {
        return [=](const ILibraryInfo::BuildOption& buildOption) {
            visitor(self, buildOption);
        };
    }

    static inline ILibraryInfo::HardwareResourceVisitor wrapHardwareResourceVisitor(
            void* self, HardwareResourceCythonVisitor visitor) {
        return [=](const ILibraryInfo::HardwareResource& hardwareResource) {
            visitor(self, hardwareResource);
        };
    }
    """

    ctypedef void (*BuildOptionCythonVisitor)(void*, const BuildOption&)

    ctypedef void (*HardwareResourceCythonVisitor)(void*, const HardwareResource&)

    BuildOptionVisitor wrapBuildOptionVisitor(void* self, BuildOptionCythonVisitor visitor)

    HardwareResourceVisitor wrapHardwareResourceVisitor(void* self, HardwareResourceCythonVisitor visitor)


cdef class CppLibraryInfo:
    
    # Attributes:

    cdef unique_ptr[ILibraryInfo] library_info_ptr

    cdef list __build_options

    cdef list __hardware_resources

    # Functions:

    cdef __visit_build_option(self, const BuildOption& build_option)

    cdef __visit_hardware_resource(self, const HardwareResource& hardware_resource)
