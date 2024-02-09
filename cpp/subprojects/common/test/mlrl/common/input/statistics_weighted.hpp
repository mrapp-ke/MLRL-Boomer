#pragma once

#include "mlrl/common/statistics/statistics_weighted.hpp"

#include <unordered_set>

/**
 * An implementation of the interface `IWeightedStatistics` for testing purposes.
 */
class WeightedStatistics final : public IWeightedStatistics {
    public:

        /**
         * A set that stores the indices of all statistics marked as covered.
         */
        std::unordered_set<uint32> coveredStatistics;

        uint32 getNumStatistics() const override {
            throw std::runtime_error("not implemented");
        }

        uint32 getNumLabels() const override {
            throw std::runtime_error("not implemented");
        }

        std::unique_ptr<IWeightedStatisticsSubset> createSubset(const CompleteIndexVector&) const override {
            throw std::runtime_error("not implemented");
        }

        std::unique_ptr<IWeightedStatisticsSubset> createSubset(const PartialIndexVector&) const override {
            throw std::runtime_error("not implemented");
        }

        std::unique_ptr<IWeightedStatistics> copy() const override {
            throw std::runtime_error("not implemented");
        }

        void resetCoveredStatistics() override {
            coveredStatistics.clear();
        }

        void addCoveredStatistic(uint32 statisticIndex) override {
            coveredStatistics.insert(statisticIndex);
        }

        void removeCoveredStatistic(uint32 statisticIndex) override {
            coveredStatistics.erase(statisticIndex);
        }

        std::unique_ptr<IHistogram> createHistogram(const DenseBinIndexVector& binIndexVector,
                                                    uint32 numBins) const override {
            throw std::runtime_error("not implemented");
        }

        std::unique_ptr<IHistogram> createHistogram(const DokBinIndexVector& binIndexVector,
                                                    uint32 numBins) const override {
            throw std::runtime_error("not implemented");
        }
};
