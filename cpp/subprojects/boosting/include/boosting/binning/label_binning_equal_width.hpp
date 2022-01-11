/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/binning/label_binning.hpp"


namespace boosting {

    /**
     * Allows to configure a method that assigns labels to bins in a way such that each bin contains labels for which
     * the predicted score is expected to belong to the same value range.
     */
    class EqualWidthLabelBinningConfig final : public ILabelBinningConfig {

        private:

            float32 binRatio_;

            uint32 minBins_;

            uint32 maxBins_;

        public:

            EqualWidthLabelBinningConfig();

            /**
             * Returns the percentage that specifies how many bins are used.
             *
             * @param The percentage that specifies how many bins are used
             */
            float32 getBinRatio() const;

            /**
             * Sets the percentage that specifies how many should be used.
             *
             * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 labels are a
             *                  available, a percentage of 0.5 means that `ceil(0.5 * 100) = 50` bins should be used.
             *                  Must be in (0, 1)
             * @return          A reference to an object of type `EqualWidthLabelBinningConfig` that allows further
             *                  configuration of the method that assigns labels to bins
             */
            EqualWidthLabelBinningConfig& setBinRatio(float32 binRatio);

            /**
             * Returns the minimum number of bins that is used.
             *
             * @return The minimum number of bins that is used
             */
            uint32 getMinBins() const;

            /**
             * Sets the minimum number of bins that should be used.
             *
             * @param minBins   The minimum number of bins that should be used. Must be at least 2
             * @return          A reference to an object of type `EqualWidthLabelBinningConfig` that allows further
             *                  configuration of the method that assigns labels to bins
             */
            EqualWidthLabelBinningConfig& setMinBins(uint32 minBins);

            /**
             * Returns the maximum number of bins that is used.
             *
             * @return The maximum number of bins that is used
             */
            uint32 getMaxBins() const;

            /**
             * Sets the maximum number of bins that should be used.
             *
             * @param maxBins   The maximum number of bins that should be used. Must be at least the minimum number of
             *                  bins or 0, if the maximum number of bins should not be restricted
             * @return          A reference to an object of type `EqualWidthLabelBinningConfig` that allows further
             *                  configuration of the method that assigns labels to bins
             */
            EqualWidthLabelBinningConfig& setMaxBins(uint32 maxBins);

    };

    /**
     * Allows to create instances of the class `EqualWidthLabelBinning` that assign labels to bins in a way such that
     * each bin contains labels for which the predicted score is expected to belong to the same value range.
     */
    class EqualWidthLabelBinningFactory final : public ILabelBinningFactory {

        private:

            float32 binRatio_;

            uint32 minBins_;

            uint32 maxBins_;

        public:

            /**
             * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 labels are a
             *                  available, a percentage of 0.5 means that `ceil(0.5 * 100) = 50` bins should be used.
             *                  Must be in (0, 1)
             * @param minBins   The minimum number of bins that should be used. Must be at least 2
             * @param maxBins   The maximum number of bins that should be used. Must be at least `minBins` or 0, if the
             *                  maximum number of bins should not be restricted
             */
            EqualWidthLabelBinningFactory(float32 binRatio, uint32 minBins, uint32 maxBins);

            std::unique_ptr<ILabelBinning> create() const override;

    };

}
