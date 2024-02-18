#include "mlrl/common/binning/bin_index_vector_dok.hpp"

DokBinIndexVector::DokBinIndexVector()
    : DokVectorDecorator<AllocatedDokVector<uint32>>(AllocatedDokVector<uint32>(BIN_INDEX_SPARSE)) {}

uint32 DokBinIndexVector::getBinIndex(uint32 exampleIndex) const {
    return this->view[exampleIndex];
}

void DokBinIndexVector::setBinIndex(uint32 exampleIndex, uint32 binIndex) {
    this->view.set(exampleIndex, binIndex);
}
