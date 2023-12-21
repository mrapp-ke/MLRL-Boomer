#include "mlrl/common/binning/bin_index_vector_dok.hpp"

#include "mlrl/common/statistics/statistics_weighted.hpp"

DokBinIndexVector::DokBinIndexVector()
    : DokVectorDecorator<AllocatedDokVector<uint32>>(AllocatedDokVector<uint32>(BIN_INDEX_SPARSE)) {}

uint32 DokBinIndexVector::getBinIndex(uint32 exampleIndex) const {
    return this->view[exampleIndex];
}

void DokBinIndexVector::setBinIndex(uint32 exampleIndex, uint32 binIndex) {
    this->view.set(exampleIndex, binIndex);
}

std::unique_ptr<IHistogram> DokBinIndexVector::createHistogram(const IWeightedStatistics& statistics,
                                                               uint32 numBins) const {
    return statistics.createHistogram(*this, numBins);
}
