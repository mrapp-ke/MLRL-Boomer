/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/dll_exports.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/**
 * The version of the file format that is used for writing model files. It is increased whenever the format is changed
 * in a way that is not compatible with previous versions.
 */
constexpr std::int64_t MODEL_FILE_FORMAT_VERSION = 1;

/**
 * Specifies the types of values that may be stored by attributes and arrays in model files.
 */
enum class ModelFileValueType : uint8 {
    INT,
    FLOAT,
    STRING
};

/**
 * Defines an interface for all classes that allow to write models and metadata to a model file. Model files are HDF5
 * files that exist in memory and can be obtained as a sequence of bytes via the function `finish`.
 *
 * Groups are specified via paths relative to the root of the file, using "/" as a separator. An empty string, or "/",
 * refers to the root of the file. Groups are created automatically when they are written to for the first time.
 *
 * All functions throw a `std::runtime_error` if an operation fails, e.g., because an array with the same name has
 * already been written, or because the function `finish` has already been called.
 */
class MLRLCOMMON_API IModelFileWriter {
    public:

        virtual ~IModelFileWriter() {}

        /**
         * Sets an attribute of a group with a string as its value. An existing attribute with the same name is
         * replaced.
         *
         * @param group The path of the group
         * @param name  The name of the attribute
         * @param value The value of the attribute
         */
        virtual void setStringAttribute(const std::string& group, const std::string& name,
                                        const std::string& value) = 0;

        /**
         * Sets an attribute of a group with an integer value. An existing attribute with the same name is replaced.
         *
         * @param group The path of the group
         * @param name  The name of the attribute
         * @param value The value of the attribute
         */
        virtual void setIntAttribute(const std::string& group, const std::string& name, std::int64_t value) = 0;

        /**
         * Sets an attribute of a group with a floating point value. An existing attribute with the same name is
         * replaced.
         *
         * @param group The path of the group
         * @param name  The name of the attribute
         * @param value The value of the attribute
         */
        virtual void setFloatAttribute(const std::string& group, const std::string& name, float64 value) = 0;

        /**
         * Writes an array of strings to a group.
         *
         * @param group     The path of the group
         * @param name      The name of the array
         * @param values    A reference to a `std::vector` that stores the strings to be written
         */
        virtual void writeStringArray(const std::string& group, const std::string& name,
                                      const std::vector<std::string>& values) = 0;

        /**
         * Writes an array of integers to a group.
         *
         * @param group         The path of the group
         * @param name          The name of the array
         * @param values        A pointer to an array that stores the values to be written. May be a null pointer, if
         *                      `numValues` is 0
         * @param numValues     The number of values to be written
         */
        virtual void writeInt64Array(const std::string& group, const std::string& name, const std::int64_t* values,
                                     std::size_t numValues) = 0;

        /**
         * Writes an array of floating point values to a group.
         *
         * @param group         The path of the group
         * @param name          The name of the array
         * @param values        A pointer to an array that stores the values to be written. May be a null pointer, if
         *                      `numValues` is 0
         * @param numValues     The number of values to be written
         */
        virtual void writeFloat64Array(const std::string& group, const std::string& name, const float64* values,
                                       std::size_t numValues) = 0;

        /**
         * Finishes writing the model file and returns its content. Afterwards, the writer cannot be used anymore.
         *
         * @return A `std::vector` that stores the content of the model file
         */
        virtual std::vector<uint8> finish() = 0;
};

/**
 * Defines an interface for all classes that allow to read models and metadata from a model file that has been
 * written via an `IModelFileWriter`.
 *
 * Groups are specified via paths relative to the root of the file, using "/" as a separator. An empty string, or "/",
 * refers to the root of the file.
 *
 * All functions throw a `std::runtime_error` if an operation fails, e.g., because a group, attribute or array does not
 * exist, or because it is not of the requested type.
 */
class MLRLCOMMON_API IModelFileReader {
    public:

        virtual ~IModelFileReader() {}

        /**
         * Returns the version of the file format that has been used for writing the model file.
         *
         * @return The version of the file format
         */
        virtual std::int64_t getFormatVersion() const = 0;

        /**
         * Returns the version of the library that has been used for writing the model file.
         *
         * @return A string that specifies the version of the library
         */
        virtual std::string getLibraryVersion() const = 0;

        /**
         * Returns whether a specific group exists or not.
         *
         * @param group The path of the group
         * @return      True, if the group exists, false otherwise
         */
        virtual bool hasGroup(const std::string& group) const = 0;

        /**
         * Returns whether a group contains an array with a specific name or not.
         *
         * @param group The path of the group
         * @param name  The name of the array
         * @return      True, if the array exists, false otherwise
         */
        virtual bool hasDataset(const std::string& group, const std::string& name) const = 0;

        /**
         * Returns whether a group has an attribute with a specific name or not.
         *
         * @param group The path of the group
         * @param name  The name of the attribute
         * @return      True, if the attribute exists, false otherwise
         */
        virtual bool hasAttribute(const std::string& group, const std::string& name) const = 0;

        /**
         * Returns the names of all attributes of a group in alphabetical order.
         *
         * @param group The path of the group
         * @return      A `std::vector` that stores the names of the attributes
         */
        virtual std::vector<std::string> getAttributeNames(const std::string& group) const = 0;

        /**
         * Returns the type of the value of an attribute.
         *
         * @param group The path of the group
         * @param name  The name of the attribute
         * @return      A value of the enum `ModelFileValueType`
         */
        virtual ModelFileValueType getAttributeType(const std::string& group, const std::string& name) const = 0;

        /**
         * Returns the value of an attribute with a string as its value.
         *
         * @param group The path of the group
         * @param name  The name of the attribute
         * @return      The value of the attribute
         */
        virtual std::string getStringAttribute(const std::string& group, const std::string& name) const = 0;

        /**
         * Returns the value of an attribute with an integer value.
         *
         * @param group The path of the group
         * @param name  The name of the attribute
         * @return      The value of the attribute
         */
        virtual std::int64_t getIntAttribute(const std::string& group, const std::string& name) const = 0;

        /**
         * Returns the value of an attribute with a floating point value.
         *
         * @param group The path of the group
         * @param name  The name of the attribute
         * @return      The value of the attribute
         */
        virtual float64 getFloatAttribute(const std::string& group, const std::string& name) const = 0;

        /**
         * Returns the type of the values stored in an array.
         *
         * @param group The path of the group
         * @param name  The name of the array
         * @return      A value of the enum `ModelFileValueType`
         */
        virtual ModelFileValueType getArrayType(const std::string& group, const std::string& name) const = 0;

        /**
         * Reads an array of strings.
         *
         * @param group The path of the group
         * @param name  The name of the array
         * @return      A `std::vector` that stores the strings
         */
        virtual std::vector<std::string> readStringArray(const std::string& group, const std::string& name) const = 0;

        /**
         * Reads an array of integers that has been written via the function `IModelFileWriter::writeInt64Array`.
         *
         * @param group The path of the group
         * @param name  The name of the array
         * @return      A `std::vector` that stores the values
         */
        virtual std::vector<std::int64_t> readInt64Array(const std::string& group, const std::string& name) const = 0;

        /**
         * Reads an array of floating point values that has been written via the function
         * `IModelFileWriter::writeFloat64Array`.
         *
         * @param group The path of the group
         * @param name  The name of the array
         * @return      A `std::vector` that stores the values
         */
        virtual std::vector<float64> readFloat64Array(const std::string& group, const std::string& name) const = 0;
};

/**
 * Creates and returns a new object of type `IModelFileWriter` that writes to a new, empty model file.
 *
 * @return An unique pointer to an object of type `IModelFileWriter` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IModelFileWriter> createModelFileWriter();

/**
 * Creates and returns a new object of type `IModelFileReader` that reads from an existing model file. The given data
 * is copied, i.e., it does not need to outlive the reader.
 *
 * @throws std::runtime_error   If the given data is not a valid model file or if its file format version is not
 *                              supported
 * @param data                  A pointer to an array that stores the content of the model file
 * @param size                  The size of the model file in bytes
 * @return                      An unique pointer to an object of type `IModelFileReader` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IModelFileReader> createModelFileReader(const uint8* data, std::size_t size);
