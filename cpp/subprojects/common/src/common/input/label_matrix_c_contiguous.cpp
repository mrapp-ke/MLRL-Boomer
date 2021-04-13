#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/statistics/statistics_provider_factory.hpp"
#include "common/sampling/instance_sampling.hpp"


CContiguousLabelMatrix::CContiguousLabelMatrix(uint32 numRows, uint32 numCols, uint8* array)
    : view_(CContiguousView<uint8>(numRows, numCols, array)) {

}

CContiguousLabelMatrix::value_const_iterator CContiguousLabelMatrix::row_values_cbegin(uint32 row) const {
    return view_.row_cbegin(row);
}

CContiguousLabelMatrix::value_const_iterator CContiguousLabelMatrix::row_values_cend(uint32 row) const {
    return view_.row_cend(row);
}

uint32 CContiguousLabelMatrix::getNumRows() const {
    return view_.getNumRows();
}

uint32 CContiguousLabelMatrix::getNumCols() const {
    return view_.getNumCols();
}

std::unique_ptr<LabelVector> CContiguousLabelMatrix::getLabelVector(uint32 row) const {
    uint32 numCols = this->getNumCols();
    std::unique_ptr<LabelVector> labelVectorPtr = std::make_unique<LabelVector>(numCols);
    LabelVector::index_iterator iterator = labelVectorPtr->indices_begin();
    value_const_iterator labelIterator = this->row_values_cbegin(row);
    uint32 n = 0;

    for (uint32 i = 0; i < numCols; i++) {
        if (labelIterator[i]) {
            iterator[n] = i;
            n++;
        }
    }

    labelVectorPtr->setNumElements(n, true);
    return labelVectorPtr;
}

std::unique_ptr<IStatisticsProvider> CContiguousLabelMatrix::createStatisticsProvider(
        const IStatisticsProviderFactory& factory) const {
    return factory.create(*this);
}

std::unique_ptr<IInstanceSubSampling> CContiguousLabelMatrix::createInstanceSubSampling(
        const IInstanceSubSamplingFactory& factory) const {
    return factory.create(*this);
}
