/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "hdf5_handle.hpp"
#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/dll_exports.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/**
 * Specifies the numeric types of elements that can be stored in HDF5 datasets.
 */
enum class Hdf5ElementType : uint8 {
    UINT8,
    UINT32,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64
};

/**
 * A type trait that maps C++ types to the corresponding `Hdf5ElementType`. Only the types `uint8`, `uint32`, `int32`,
 * `std::int64_t`, `float32` and `float64` are supported.
 *
 * @tparam T The C++ type
 */
template<typename T>
struct Hdf5ElementTypeOf;

template<>
struct Hdf5ElementTypeOf<uint8> {
        static constexpr Hdf5ElementType value = Hdf5ElementType::UINT8;
};

template<>
struct Hdf5ElementTypeOf<uint32> {
        static constexpr Hdf5ElementType value = Hdf5ElementType::UINT32;
};

template<>
struct Hdf5ElementTypeOf<int32> {
        static constexpr Hdf5ElementType value = Hdf5ElementType::INT32;
};

template<>
struct Hdf5ElementTypeOf<std::int64_t> {
        static constexpr Hdf5ElementType value = Hdf5ElementType::INT64;
};

template<>
struct Hdf5ElementTypeOf<float32> {
        static constexpr Hdf5ElementType value = Hdf5ElementType::FLOAT32;
};

template<>
struct Hdf5ElementTypeOf<float64> {
        static constexpr Hdf5ElementType value = Hdf5ElementType::FLOAT64;
};

/**
 * Specifies the classes of data types that may be used by HDF5 attributes and datasets.
 */
enum class Hdf5TypeClass : uint8 {
    INTEGER,
    FLOAT,
    STRING,
    OTHER
};

/**
 * Provides access to a group in an HDF5 file, which may contain other groups, datasets and attributes. Datasets are
 * always one-dimensional, attributes are always scalar.
 *
 * Paths of groups and datasets are specified relative to this group, using "/" as a separator. A group must not be
 * used after the file it belongs to has been destroyed.
 *
 * All methods throw a `std::runtime_error` if an operation fails, e.g., because a group, dataset or attribute does
 * not exist, or because its data type does not match the expected one.
 */
class MLRLCOMMON_API Hdf5Group final {
    private:

        Hdf5Handle handle_;

        void writeDataset(const std::string& path, Hdf5ElementType elementType, const void* data,
                          std::size_t numElements);

        void readDataset(const std::string& path, Hdf5ElementType elementType, void* data,
                         std::size_t numElements) const;

    public:

        /**
         * @param handle The handle of the HDF5 group, which should be owned by this object
         */
        explicit Hdf5Group(Hdf5Handle&& handle);

        /**
         * Creates a new group. Intermediate groups are created if necessary.
         *
         * @param path  The path of the group to be created
         * @return      An object of type `Hdf5Group` that provides access to the new group
         */
        Hdf5Group createGroup(const std::string& path);

        /**
         * Opens an existing group.
         *
         * @param path  The path of the group to be opened
         * @return      An object of type `Hdf5Group` that provides access to the group
         */
        Hdf5Group openGroup(const std::string& path) const;

        /**
         * Returns whether a group with a specific path exists or not.
         *
         * @param path  The path of the group
         * @return      True, if the group exists, false otherwise
         */
        bool hasGroup(const std::string& path) const;

        /**
         * Returns whether a dataset with a specific path exists or not.
         *
         * @param path  The path of the dataset
         * @return      True, if the dataset exists, false otherwise
         */
        bool hasDataset(const std::string& path) const;

        /**
         * Writes a new one-dimensional dataset with numeric elements. Intermediate groups are created if necessary.
         *
         * @tparam T            The type of the elements
         * @param path          The path of the dataset
         * @param data          A pointer to an array that stores the elements to be written. May be a null pointer,
         *                      if `numElements` is 0
         * @param numElements   The number of elements to be written. May be 0
         */
        template<typename T>
        void writeDataset(const std::string& path, const T* data, std::size_t numElements) {
            this->writeDataset(path, Hdf5ElementTypeOf<T>::value, static_cast<const void*>(data), numElements);
        }

        /**
         * Reads an existing one-dimensional dataset with numeric elements. The elements must have been stored using
         * exactly the requested type, i.e., no type conversions are performed.
         *
         * @tparam T    The type of the elements
         * @param path  The path of the dataset
         * @return      A `std::vector` that stores the elements of the dataset
         */
        template<typename T>
        std::vector<T> readDataset(const std::string& path) const {
            std::vector<T> result(this->getDatasetSize(path));
            this->readDataset(path, Hdf5ElementTypeOf<T>::value, static_cast<void*>(result.data()), result.size());
            return result;
        }

        /**
         * Writes a new one-dimensional dataset with variable-length UTF-8 strings as elements. Intermediate groups are
         * created if necessary.
         *
         * @param path      The path of the dataset
         * @param values    A reference to a `std::vector` that stores the strings to be written. May be empty
         */
        void writeStringDataset(const std::string& path, const std::vector<std::string>& values);

        /**
         * Reads an existing one-dimensional dataset with variable-length strings as elements.
         *
         * @param path  The path of the dataset
         * @return      A `std::vector` that stores the strings
         */
        std::vector<std::string> readStringDataset(const std::string& path) const;

        /**
         * Returns the number of elements in an existing one-dimensional dataset.
         *
         * @param path  The path of the dataset
         * @return      The number of elements in the dataset
         */
        std::size_t getDatasetSize(const std::string& path) const;

        /**
         * Returns the class of the data type that is used by an existing dataset.
         *
         * @param path  The path of the dataset
         * @return      A value of the enum `Hdf5TypeClass`
         */
        Hdf5TypeClass getDatasetTypeClass(const std::string& path) const;

        /**
         * Sets an attribute of this group with an integer value. An existing attribute with the same name is replaced.
         *
         * @param name  The name of the attribute
         * @param value The value of the attribute
         */
        void setIntAttribute(const std::string& name, std::int64_t value);

        /**
         * Sets an attribute of this group with a floating point value. An existing attribute with the same name is
         * replaced.
         *
         * @param name  The name of the attribute
         * @param value The value of the attribute
         */
        void setFloatAttribute(const std::string& name, float64 value);

        /**
         * Sets an attribute of this group with a variable-length UTF-8 string as its value. An existing attribute with
         * the same name is replaced.
         *
         * @param name  The name of the attribute
         * @param value The value of the attribute
         */
        void setStringAttribute(const std::string& name, const std::string& value);

        /**
         * Returns whether this group has an attribute with a specific name or not.
         *
         * @param name  The name of the attribute
         * @return      True, if the attribute exists, false otherwise
         */
        bool hasAttribute(const std::string& name) const;

        /**
         * Returns the names of all attributes of this group in alphabetical order.
         *
         * @return A `std::vector` that stores the names of the attributes
         */
        std::vector<std::string> getAttributeNames() const;

        /**
         * Returns the class of the data type that is used by an existing attribute of this group.
         *
         * @param name  The name of the attribute
         * @return      A value of the enum `Hdf5TypeClass`
         */
        Hdf5TypeClass getAttributeTypeClass(const std::string& name) const;

        /**
         * Returns the value of an existing attribute of this group with an integer value.
         *
         * @param name  The name of the attribute
         * @return      The value of the attribute
         */
        std::int64_t getIntAttribute(const std::string& name) const;

        /**
         * Returns the value of an existing attribute of this group with a floating point value.
         *
         * @param name  The name of the attribute
         * @return      The value of the attribute
         */
        float64 getFloatAttribute(const std::string& name) const;

        /**
         * Returns the value of an existing attribute of this group with a variable-length string as its value.
         *
         * @param name  The name of the attribute
         * @return      The value of the attribute
         */
        std::string getStringAttribute(const std::string& name) const;
};
