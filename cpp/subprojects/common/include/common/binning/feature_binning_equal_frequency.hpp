/*
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/feature_binning.hpp"


/**
 * Allows to configure a method that assigns numerical feature values to bins, such that each bins contains
 * approximately the same number of values.
 */
class EqualFrequencyFeatureBinningConfig : public IFeatureBinningConfig {

    private:

        float32 binRatio_;

        uint32 minBins_;

        uint32 maxBins_;

    public:

        EqualFrequencyFeatureBinningConfig();

        /**
         * Returns the percentage that specifies how many bins are used.
         *
         * @return The percentage that specifies how many bins are used
         */
        float32 getBinRatio() const;

        /**
         * Sets the percentage that specifies how many bins should be used.
         *
         * @param binRatio  The percentage that specifies how many bins should be used, e.g., if 100 values are
         *                  available, a percentage of 0.5 means that `ceil(0.5 * 100) = 50` bins should be used. Must
         *                  be in (0, 1)
         * @return          A reference to an object of type `EqualFrequencyFeatureBinningConfig` that allows further
         *                  configuration of the method that assigns numerical feature values to bins
         */
        EqualFrequencyFeatureBinningConfig& setBinRatio(float32 binRatio);

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
         * @return          A reference to an object of type `EqualFrequencyFeatureBinningConfig` that allows further
         *                  configuration of the method that assigns numerical feature values to bins
         */
        EqualFrequencyFeatureBinningConfig& setMinBins(uint32 minBins);

        /**
         * Returns the maximum number of bins that is used.
         *
         * @return The maximum number of bins that is used
         */
        uint32 getMaxBins() const;

        /**
         * Sets the maximum number of bins that should be used.
         *
         * @param maxBins   The maximum number of bins that should be used. Must be at least the minimum number of bins
         *                  or 0, if the maximum number of bins should not be restricted
         * @return          A reference to an object of type `EqualFrequencyFeatureBinningConfig` that allows further
         *                  configuration of the method that assigns numerical feature values to bins
         */
        EqualFrequencyFeatureBinningConfig& setMaxBins(uint32 maxBins);

};

/**
 * Allows to create instances of the type `IFeatureBinning` that assign numerical feature values to bins, such that each
 * bin contains approximately the same number of values.
 */
class EqualFrequencyFeatureBinningFactory final : public IFeatureBinningFactory {

    private:

        float32 binRatio_;

        uint32 minBins_;

        uint32 maxBins_;

    public:

        /**
         * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 values are available,
         *                  a percentage of 0.5 means that `ceil(0.5 * 100) = 50` bins should be used. Must be in (0, 1)
         * @param minBins   The minimum number of bins to be used. Must be at least 2
         * @param maxBins   The maximum number of bins to be used. Must be at least `minBins` or 0, if the maximum
         *                  number of bins should not be restricted
         */
        EqualFrequencyFeatureBinningFactory(float32 binRatio, uint32 minBins, uint32 maxBins);

        std::unique_ptr<IFeatureBinning> create() const override;

};
