from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from mlrl.common.cython._types cimport float64, uint8
from mlrl.common.cython.output_space_info cimport IOutputSpaceInfo
from mlrl.common.cython.probability_calibration cimport IJointProbabilityCalibrationModel, \
    IMarginalProbabilityCalibrationModel
from mlrl.common.cython.rule_model cimport IRuleModel


cdef extern from "mlrl/common/serialization/model_file.hpp" nogil:

    int64_t MODEL_FILE_FORMAT_VERSION

    cdef enum class ModelFileValueType:
        INT
        FLOAT
        STRING

    cdef cppclass IModelFileWriter:

        # Functions:

        void setStringAttribute(const string& group, const string& name, const string& value) except +

        void setIntAttribute(const string& group, const string& name, int64_t value) except +

        void setFloatAttribute(const string& group, const string& name, float64 value) except +

        void writeStringArray(const string& group, const string& name, const vector[string]& values) except +

        void writeInt64Array(const string& group, const string& name, const int64_t* values,
                             size_t numValues) except +

        void writeFloat64Array(const string& group, const string& name, const float64* values,
                               size_t numValues) except +

        void writeMarginalProbabilityCalibrationModel(const IMarginalProbabilityCalibrationModel& model) except +

        void writeJointProbabilityCalibrationModel(const IJointProbabilityCalibrationModel& model) except +

        void writeOutputSpaceInfo(const IOutputSpaceInfo& outputSpaceInfo) except +

        void writeRuleModel(const IRuleModel& ruleModel) except +

        vector[uint8] finish() except +

    cdef cppclass IModelFileReader:

        # Functions:

        int64_t getFormatVersion() except +

        string getLibraryVersion() except +

        bool hasGroup(const string& group) except +

        bool hasDataset(const string& group, const string& name) except +

        bool hasAttribute(const string& group, const string& name) except +

        vector[string] getAttributeNames(const string& group) except +

        ModelFileValueType getAttributeType(const string& group, const string& name) except +

        string getStringAttribute(const string& group, const string& name) except +

        int64_t getIntAttribute(const string& group, const string& name) except +

        float64 getFloatAttribute(const string& group, const string& name) except +

        ModelFileValueType getArrayType(const string& group, const string& name) except +

        vector[string] readStringArray(const string& group, const string& name) except +

        vector[int64_t] readInt64Array(const string& group, const string& name) except +

        vector[float64] readFloat64Array(const string& group, const string& name) except +

        bool hasMarginalProbabilityCalibrationModel() except +

        unique_ptr[IMarginalProbabilityCalibrationModel] readMarginalProbabilityCalibrationModel() except +

        bool hasJointProbabilityCalibrationModel() except +

        unique_ptr[IJointProbabilityCalibrationModel] readJointProbabilityCalibrationModel() except +

        bool hasOutputSpaceInfo() except +

        unique_ptr[IOutputSpaceInfo] readOutputSpaceInfo() except +

        bool hasRuleModel() except +

        unique_ptr[IRuleModel] readRuleModel() except +

    unique_ptr[IModelFileWriter] createModelFileWriter() except +

    unique_ptr[IModelFileReader] createModelFileReader(const uint8* data, size_t size) except +


cdef class ModelFileWriter:

    # Attributes:

    cdef unique_ptr[IModelFileWriter] writer_ptr


cdef class ModelFileReader:

    # Attributes:

    cdef unique_ptr[IModelFileReader] reader_ptr
