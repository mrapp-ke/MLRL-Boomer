#include "hdf5_file.hpp"

#include "hdf5_error.hpp"

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

/**
 * The number of bytes by which the memory of an in-memory file is increased when needed.
 */
static constexpr std::size_t MEMORY_INCREMENT = 64 * 1024;

/**
 * Returns a unique name for an in-memory file. The HDF5 library identifies in-memory files that are not backed by a
 * file on disk by their names, i.e., files that are open at the same time must use different names.
 *
 * @return A unique name
 */
static std::string createUniqueFileName() {
    static std::atomic<std::uint64_t> counter(0);
    return "mlrl_in_memory_file_" + std::to_string(counter.fetch_add(1)) + ".h5";
}

static Hdf5Handle createFileAccessPropertyList() {
    Hdf5Handle fileAccessPropertyList(
      checkHdf5(H5Pcreate(H5P_FILE_ACCESS), "Failed to create file access property list"), &H5Pclose);
    checkHdf5(H5Pset_fapl_core(fileAccessPropertyList.get(), MEMORY_INCREMENT, false),
              "Failed to configure in-memory file driver");
    return fileAccessPropertyList;
}

Hdf5File::Hdf5File(Hdf5Handle&& handle) : handle_(std::move(handle)) {}

Hdf5File Hdf5File::createInMemory() {
    initializeHdf5();
    Hdf5Handle fileAccessPropertyList = createFileAccessPropertyList();
    hid_t fileId =
      checkHdf5(H5Fcreate(createUniqueFileName().c_str(), H5F_ACC_EXCL, H5P_DEFAULT, fileAccessPropertyList.get()),
                "Failed to create in-memory file");
    return Hdf5File(Hdf5Handle(fileId, &H5Fclose));
}

Hdf5File Hdf5File::openFromImage(const uint8* data, std::size_t size) {
    if (!data || size == 0) {
        throw std::runtime_error("Failed to open HDF5 file from image: The image is empty");
    }

    initializeHdf5();
    Hdf5Handle fileAccessPropertyList = createFileAccessPropertyList();
    // The data is copied by the HDF5 library, i.e., it is never modified
    checkHdf5(H5Pset_file_image(fileAccessPropertyList.get(), const_cast<uint8*>(data), size),
              "Failed to set file image");
    hid_t fileId = checkHdf5(H5Fopen(createUniqueFileName().c_str(), H5F_ACC_RDONLY, fileAccessPropertyList.get()),
                             "Failed to open file from image (the data is probably not a valid HDF5 file)");
    return Hdf5File(Hdf5Handle(fileId, &H5Fclose));
}

std::vector<uint8> Hdf5File::getImage() {
    hid_t fileId = handle_.get();
    checkHdf5(H5Fflush(fileId, H5F_SCOPE_GLOBAL), "Failed to flush file");
    ssize_t size = checkHdf5(H5Fget_file_image(fileId, nullptr, 0), "Failed to obtain size of file image");
    std::vector<uint8> image(static_cast<std::size_t>(size));

    if (size > 0) {
        checkHdf5(H5Fget_file_image(fileId, image.data(), image.size()), "Failed to obtain file image");
    }

    return image;
}

Hdf5Group Hdf5File::getRootGroup() const {
    hid_t groupId = checkHdf5(H5Gopen2(handle_.get(), "/", H5P_DEFAULT), "Failed to open root group");
    return Hdf5Group(Hdf5Handle(groupId, &H5Gclose));
}
