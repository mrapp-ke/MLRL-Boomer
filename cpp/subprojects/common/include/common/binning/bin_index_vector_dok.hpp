/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/binning/bin_index_vector.hpp"
#include "common/data/vector_dok.hpp"


/**
 * Stores the indices of the bins, individual examples have been assigned to, using the dictionaries of keys (DOK)
 * format.
 */
class DokBinIndexVector final : public IBinIndexVector {

    private:

        DokVector<uint32> vector_;

    public:

        DokBinIndexVector();

        uint32 getBinIndex(uint32 exampleIndex) const override;

        void setBinIndex(uint32 exampleIndex, uint32 binIndex) override;

};
