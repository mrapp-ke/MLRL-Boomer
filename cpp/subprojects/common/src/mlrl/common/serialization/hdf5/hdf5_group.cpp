#include "hdf5_group.hpp"

#include "hdf5_error.hpp"

#include <stdexcept>
#include <utility>

static hid_t getNativeType(Hdf5ElementType elementType) {
    switch (elementType) {
        case Hdf5ElementType::UINT8:
            return H5T_NATIVE_UINT8;
        case Hdf5ElementType::UINT32:
            return H5T_NATIVE_UINT32;
        case Hdf5ElementType::INT32:
            return H5T_NATIVE_INT32;
        case Hdf5ElementType::INT64:
            return H5T_NATIVE_INT64;
        case Hdf5ElementType::FLOAT32:
            return H5T_NATIVE_FLOAT;
        case Hdf5ElementType::FLOAT64:
            return H5T_NATIVE_DOUBLE;
        default:
            throw std::invalid_argument("Unsupported element type");
    }
}

static Hdf5TypeClass getTypeClass(hid_t typeId) {
    switch (checkHdf5(H5Tget_class(typeId), "Failed to obtain class of data type")) {
        case H5T_INTEGER:
            return Hdf5TypeClass::INTEGER;
        case H5T_FLOAT:
            return Hdf5TypeClass::FLOAT;
        case H5T_STRING:
            return Hdf5TypeClass::STRING;
        default:
            return Hdf5TypeClass::OTHER;
    }
}

static Hdf5Handle createVariableLengthStringType() {
    Hdf5Handle typeHandle(checkHdf5(H5Tcopy(H5T_C_S1), "Failed to create string type"), &H5Tclose);
    checkHdf5(H5Tset_size(typeHandle.get(), H5T_VARIABLE), "Failed to create variable-length string type");
    checkHdf5(H5Tset_cset(typeHandle.get(), H5T_CSET_UTF8), "Failed to set character set of string type");
    return typeHandle;
}

static Hdf5Handle createIntermediateGroupLinkPropertyList() {
    Hdf5Handle propertyList(checkHdf5(H5Pcreate(H5P_LINK_CREATE), "Failed to create link creation property list"),
                            &H5Pclose);
    checkHdf5(H5Pset_create_intermediate_group(propertyList.get(), 1),
              "Failed to enable creation of intermediate groups");
    return propertyList;
}

static Hdf5Handle createOneDimensionalDataspace(std::size_t numElements) {
    hsize_t dimensions[1] = {static_cast<hsize_t>(numElements)};
    return Hdf5Handle(checkHdf5(H5Screate_simple(1, dimensions, nullptr), "Failed to create dataspace"), &H5Sclose);
}

static Hdf5Handle createScalarDataspace() {
    return Hdf5Handle(checkHdf5(H5Screate(H5S_SCALAR), "Failed to create scalar dataspace"), &H5Sclose);
}

static Hdf5Handle openDataset(hid_t groupId, const std::string& path) {
    return Hdf5Handle(
      checkHdf5(H5Dopen2(groupId, path.c_str(), H5P_DEFAULT), "Failed to open dataset \"" + path + "\""), &H5Dclose);
}

static Hdf5Handle createDataset(hid_t groupId, const std::string& path, hid_t typeId, std::size_t numElements) {
    Hdf5Handle dataspace = createOneDimensionalDataspace(numElements);
    Hdf5Handle linkCreationPropertyList = createIntermediateGroupLinkPropertyList();
    return Hdf5Handle(checkHdf5(H5Dcreate2(groupId, path.c_str(), typeId, dataspace.get(),
                                           linkCreationPropertyList.get(), H5P_DEFAULT, H5P_DEFAULT),
                                "Failed to create dataset \"" + path + "\""),
                      &H5Dclose);
}

static std::size_t getNumElements(hid_t datasetId, const std::string& path) {
    Hdf5Handle dataspace(checkHdf5(H5Dget_space(datasetId), "Failed to obtain dataspace of dataset \"" + path + "\""),
                         &H5Sclose);
    int rank =
      checkHdf5(H5Sget_simple_extent_ndims(dataspace.get()), "Failed to obtain rank of dataset \"" + path + "\"");

    if (rank != 1) {
        throw std::runtime_error("Dataset \"" + path + "\" must be one-dimensional, but has rank "
                                 + std::to_string(rank));
    }

    hsize_t dimensions[1];
    checkHdf5(H5Sget_simple_extent_dims(dataspace.get(), dimensions, nullptr),
              "Failed to obtain dimensions of dataset \"" + path + "\"");
    return static_cast<std::size_t>(dimensions[0]);
}

static Hdf5Handle getDatasetType(hid_t datasetId, const std::string& path) {
    return Hdf5Handle(checkHdf5(H5Dget_type(datasetId), "Failed to obtain data type of dataset \"" + path + "\""),
                      &H5Tclose);
}

static void checkIsVariableLengthString(hid_t typeId, const std::string& description) {
    if (getTypeClass(typeId) != Hdf5TypeClass::STRING
        || checkHdf5(H5Tis_variable_str(typeId), "Failed to inspect string type of " + description) <= 0) {
        throw std::runtime_error("Expected " + description + " to store variable-length strings");
    }
}

static std::string toStringAndFree(char* value) {
    if (!value) {
        return std::string();
    }

    std::string result;

    try {
        result = value;
    } catch (...) {
        H5free_memory(value);
        throw;
    }

    H5free_memory(value);
    return result;
}

static Hdf5Handle openAttribute(hid_t groupId, const std::string& name) {
    return Hdf5Handle(
      checkHdf5(H5Aopen(groupId, name.c_str(), H5P_DEFAULT), "Failed to open attribute \"" + name + "\""), &H5Aclose);
}

static Hdf5Handle openAttributeOfTypeClass(hid_t groupId, const std::string& name, Hdf5TypeClass expectedTypeClass,
                                           const char* expectedTypeName) {
    Hdf5Handle attribute = openAttribute(groupId, name);
    Hdf5Handle type(checkHdf5(H5Aget_type(attribute.get()), "Failed to obtain data type of attribute \"" + name + "\""),
                    &H5Tclose);

    if (getTypeClass(type.get()) != expectedTypeClass) {
        throw std::runtime_error("Expected attribute \"" + name + "\" to be of type " + expectedTypeName);
    }

    Hdf5Handle dataspace(
      checkHdf5(H5Aget_space(attribute.get()), "Failed to obtain dataspace of attribute \"" + name + "\""), &H5Sclose);

    if (checkHdf5(H5Sget_simple_extent_npoints(dataspace.get()),
                  "Failed to obtain number of elements of attribute \"" + name + "\"")
        != 1) {
        throw std::runtime_error("Expected attribute \"" + name + "\" to be scalar");
    }

    return attribute;
}

static void writeScalarAttribute(hid_t groupId, const std::string& name, hid_t typeId, const void* value) {
    if (checkHdf5(H5Aexists(groupId, name.c_str()), "Failed to check if attribute \"" + name + "\" exists") > 0) {
        checkHdf5(H5Adelete(groupId, name.c_str()), "Failed to delete existing attribute \"" + name + "\"");
    }

    Hdf5Handle dataspace = createScalarDataspace();
    Hdf5Handle attribute(checkHdf5(H5Acreate2(groupId, name.c_str(), typeId, dataspace.get(), H5P_DEFAULT, H5P_DEFAULT),
                                   "Failed to create attribute \"" + name + "\""),
                         &H5Aclose);
    checkHdf5(H5Awrite(attribute.get(), typeId, value), "Failed to write attribute \"" + name + "\"");
}

/**
 * Returns the type of the object with a specific path, relative to a group, or `H5I_BADID`, if no such object exists.
 * Intermediate objects on the path must be groups.
 */
static H5I_type_t getObjectType(hid_t groupId, const std::string& path) {
    std::size_t start = 0;

    while (start <= path.size()) {
        std::size_t end = path.find('/', start);

        if (end == std::string::npos) {
            end = path.size();
        }

        if (end > start) {
            std::string prefix = path.substr(0, end);

            if (checkHdf5(H5Lexists(groupId, prefix.c_str(), H5P_DEFAULT),
                          "Failed to check if \"" + prefix + "\" exists")
                <= 0) {
                return H5I_BADID;
            }

            Hdf5Handle object(
              checkHdf5(H5Oopen(groupId, prefix.c_str(), H5P_DEFAULT), "Failed to open object \"" + prefix + "\""),
              &H5Oclose);
            H5I_type_t objectType = H5Iget_type(object.get());

            if (end == path.size()) {
                return objectType;
            } else if (objectType != H5I_GROUP) {
                return H5I_BADID;
            }
        }

        start = end + 1;
    }

    return H5I_BADID;
}

Hdf5Group::Hdf5Group(Hdf5Handle&& handle) : handle_(std::move(handle)) {}

Hdf5Group Hdf5Group::createGroup(const std::string& path) {
    Hdf5Handle linkCreationPropertyList = createIntermediateGroupLinkPropertyList();
    hid_t groupId =
      checkHdf5(H5Gcreate2(handle_.get(), path.c_str(), linkCreationPropertyList.get(), H5P_DEFAULT, H5P_DEFAULT),
                "Failed to create group \"" + path + "\"");
    return Hdf5Group(Hdf5Handle(groupId, &H5Gclose));
}

Hdf5Group Hdf5Group::openGroup(const std::string& path) const {
    hid_t groupId =
      checkHdf5(H5Gopen2(handle_.get(), path.c_str(), H5P_DEFAULT), "Failed to open group \"" + path + "\"");
    return Hdf5Group(Hdf5Handle(groupId, &H5Gclose));
}

bool Hdf5Group::hasGroup(const std::string& path) const {
    return getObjectType(handle_.get(), path) == H5I_GROUP;
}

bool Hdf5Group::hasDataset(const std::string& path) const {
    return getObjectType(handle_.get(), path) == H5I_DATASET;
}

void Hdf5Group::writeDataset(const std::string& path, Hdf5ElementType elementType, const void* data,
                             std::size_t numElements) {
    hid_t typeId = getNativeType(elementType);
    Hdf5Handle dataset = createDataset(handle_.get(), path, typeId, numElements);

    if (numElements > 0) {
        checkHdf5(H5Dwrite(dataset.get(), typeId, H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
                  "Failed to write dataset \"" + path + "\"");
    }
}

void Hdf5Group::readDataset(const std::string& path, Hdf5ElementType elementType, void* data,
                            std::size_t numElements) const {
    hid_t typeId = getNativeType(elementType);
    Hdf5Handle dataset = openDataset(handle_.get(), path);
    Hdf5Handle storedType = getDatasetType(dataset.get(), path);
    Hdf5Handle nativeStoredType(checkHdf5(H5Tget_native_type(storedType.get(), H5T_DIR_ASCEND),
                                          "Failed to obtain native data type of dataset \"" + path + "\""),
                                &H5Tclose);

    // Type conversions are not allowed to prevent loss of precision or overflows
    if (checkHdf5(H5Tequal(nativeStoredType.get(), typeId), "Failed to compare data types") <= 0) {
        throw std::runtime_error("Dataset \"" + path + "\" does not store elements of the expected type");
    }

    if (getNumElements(dataset.get(), path) != numElements) {
        throw std::runtime_error("Dataset \"" + path + "\" does not have the expected number of elements");
    }

    if (numElements > 0) {
        checkHdf5(H5Dread(dataset.get(), typeId, H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
                  "Failed to read dataset \"" + path + "\"");
    }
}

void Hdf5Group::writeStringDataset(const std::string& path, const std::vector<std::string>& values) {
    Hdf5Handle stringType = createVariableLengthStringType();
    Hdf5Handle dataset = createDataset(handle_.get(), path, stringType.get(), values.size());

    if (!values.empty()) {
        std::vector<const char*> pointers;
        pointers.reserve(values.size());

        for (const std::string& value : values) {
            pointers.push_back(value.c_str());
        }

        checkHdf5(H5Dwrite(dataset.get(), stringType.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, pointers.data()),
                  "Failed to write dataset \"" + path + "\"");
    }
}

std::vector<std::string> Hdf5Group::readStringDataset(const std::string& path) const {
    Hdf5Handle dataset = openDataset(handle_.get(), path);
    checkIsVariableLengthString(getDatasetType(dataset.get(), path).get(), "dataset \"" + path + "\"");
    std::size_t numElements = getNumElements(dataset.get(), path);
    std::vector<std::string> result;

    if (numElements > 0) {
        Hdf5Handle stringType = createVariableLengthStringType();
        std::vector<char*> pointers(numElements, nullptr);
        checkHdf5(H5Dread(dataset.get(), stringType.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, pointers.data()),
                  "Failed to read dataset \"" + path + "\"");
        result.reserve(numElements);
        std::size_t i = 0;

        try {
            for (; i < numElements; i++) {
                result.push_back(toStringAndFree(pointers[i]));
            }
        } catch (...) {
            for (i = i + 1; i < numElements; i++) {
                H5free_memory(pointers[i]);
            }

            throw;
        }
    }

    return result;
}

std::size_t Hdf5Group::getDatasetSize(const std::string& path) const {
    Hdf5Handle dataset = openDataset(handle_.get(), path);
    return getNumElements(dataset.get(), path);
}

Hdf5TypeClass Hdf5Group::getDatasetTypeClass(const std::string& path) const {
    Hdf5Handle dataset = openDataset(handle_.get(), path);
    return getTypeClass(getDatasetType(dataset.get(), path).get());
}

void Hdf5Group::setIntAttribute(const std::string& name, std::int64_t value) {
    writeScalarAttribute(handle_.get(), name, H5T_NATIVE_INT64, &value);
}

void Hdf5Group::setFloatAttribute(const std::string& name, float64 value) {
    writeScalarAttribute(handle_.get(), name, H5T_NATIVE_DOUBLE, &value);
}

void Hdf5Group::setStringAttribute(const std::string& name, const std::string& value) {
    Hdf5Handle stringType = createVariableLengthStringType();
    const char* pointer = value.c_str();
    writeScalarAttribute(handle_.get(), name, stringType.get(), &pointer);
}

bool Hdf5Group::hasAttribute(const std::string& name) const {
    return checkHdf5(H5Aexists(handle_.get(), name.c_str()), "Failed to check if attribute \"" + name + "\" exists")
           > 0;
}

static herr_t visitAttribute(hid_t, const char* name, const H5A_info_t*, void* clientData) {
    std::vector<std::string>* names = static_cast<std::vector<std::string>*>(clientData);

    try {
        names->emplace_back(name);
    } catch (...) {
        return -1;
    }

    return 0;
}

std::vector<std::string> Hdf5Group::getAttributeNames() const {
    std::vector<std::string> names;
    hsize_t index = 0;
    checkHdf5(H5Aiterate2(handle_.get(), H5_INDEX_NAME, H5_ITER_INC, &index, &visitAttribute, &names),
              "Failed to iterate attributes");
    return names;
}

Hdf5TypeClass Hdf5Group::getAttributeTypeClass(const std::string& name) const {
    Hdf5Handle attribute = openAttribute(handle_.get(), name);
    Hdf5Handle type(checkHdf5(H5Aget_type(attribute.get()), "Failed to obtain data type of attribute \"" + name + "\""),
                    &H5Tclose);
    return getTypeClass(type.get());
}

std::int64_t Hdf5Group::getIntAttribute(const std::string& name) const {
    Hdf5Handle attribute = openAttributeOfTypeClass(handle_.get(), name, Hdf5TypeClass::INTEGER, "integer");
    std::int64_t value;
    checkHdf5(H5Aread(attribute.get(), H5T_NATIVE_INT64, &value), "Failed to read attribute \"" + name + "\"");
    return value;
}

float64 Hdf5Group::getFloatAttribute(const std::string& name) const {
    Hdf5Handle attribute = openAttributeOfTypeClass(handle_.get(), name, Hdf5TypeClass::FLOAT, "float");
    float64 value;
    checkHdf5(H5Aread(attribute.get(), H5T_NATIVE_DOUBLE, &value), "Failed to read attribute \"" + name + "\"");
    return value;
}

std::string Hdf5Group::getStringAttribute(const std::string& name) const {
    Hdf5Handle attribute = openAttributeOfTypeClass(handle_.get(), name, Hdf5TypeClass::STRING, "string");
    Hdf5Handle storedType(
      checkHdf5(H5Aget_type(attribute.get()), "Failed to obtain data type of attribute \"" + name + "\""), &H5Tclose);
    checkIsVariableLengthString(storedType.get(), "attribute \"" + name + "\"");
    Hdf5Handle stringType = createVariableLengthStringType();
    char* value = nullptr;
    checkHdf5(H5Aread(attribute.get(), stringType.get(), &value), "Failed to read attribute \"" + name + "\"");
    return toStringAndFree(value);
}
