/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_indexed_lists.hpp"
#include "mlrl/common/prediction/probability_calibration_joint.hpp"

#include <functional>
#include <memory>
#include <utility>

/**
 * Defines an interface for all models for the calibration of marginal or joint probabilities via isotonic regression.
 */
class MLRLCOMMON_API IIsotonicProbabilityCalibrationModel : public IMarginalProbabilityCalibrationModel,
                                                            public IJointProbabilityCalibrationModel {
    public:

        virtual ~IIsotonicProbabilityCalibrationModel() override {}

        /**
         * A visitor function for handling individual bins.
         */
        using BinVisitor = std::function<void(uint32 listIndex, float64 threshold, float64 probability)>;

        /**
         * Returns the number of available list of bins.
         *
         * @return The number of available list of bins
         */
        virtual uint32 getNumBinLists() const = 0;

        /**
         * Adds a new bin to the calibration model.
         *
         * @param listIndex     The index of the list, the bin should be added to
         * @param threshold     The threshold of the bin
         * @param probability   The probability of the bin
         */
        virtual void addBin(uint32 listIndex, float64 threshold, float64 probability) = 0;

        /**
         * Invokes the given visitor function for each bin that is contained by the calibration model.
         *
         * @param visitor The visitor function for handling individual bins
         */
        virtual void visit(BinVisitor visitor) const = 0;
};

/**
 * A model for the calibration of marginal or joint probabilities via isotonic regression.
 */
class IsotonicProbabilityCalibrationModel final
    : public ViewDecorator<AllocatedIndexedListsView<std::pair<float64, float64>>>,
      public IIsotonicProbabilityCalibrationModel {
    public:

        /**
         * @param numLists The total number of lists for storing bins
         */
        IsotonicProbabilityCalibrationModel(uint32 numLists);

        /**
         * Provides read-only access to the bins that correspond to a specific label.
         */
        using const_bin_list = IndexedListsView<std::pair<float64, float64>>::const_list;

        /**
         * Provides access to the bins that correspond to a specific label and allows to modify them.
         */
        using bin_list = IndexedListsView<std::pair<float64, float64>>::list;

        /**
         * Returns a list that stores the bins that correspond to a specific label.
         *
         * @param labelIndex  The index of the label
         * @return            A `const_bin_list`
         */
        const_bin_list operator[](uint32 labelIndex) const;

        /**
         * Returns a list that stores the bins that correspond to a specific label and allows to modify them.
         *
         * @param labelIndex  The index of the label
         * @return            A `bin_list`
         */
        bin_list operator[](uint32 labelIndex);

        /**
         * Fits the isotonic calibration model.
         */
        void fit();

        float64 calibrateMarginalProbability(uint32 labelIndex, float64 marginalProbability) const override;

        float64 calibrateJointProbability(uint32 labelVectorIndex, float64 jointProbability) const override;

        uint32 getNumBinLists() const override;

        void addBin(uint32 listIndex, float64 threshold, float64 probability) override;

        void visit(BinVisitor visitor) const override;
};

/**
 * Creates and returns a new object of the type `IIsotonicProbabilityCalibrationModel`.
 *
 * @param numLists  The total number of lists for storing bins
 * @return          An unique pointer to an object of type `IIsotonicProbabilityCalibrationModel` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IIsotonicProbabilityCalibrationModel> createIsotonicProbabilityCalibrationModel(
  uint32 numLists);
