/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/list_of_lists.hpp"
#include "common/data/tuple.hpp"
#include "common/prediction/probability_calibration_joint.hpp"

#include <functional>

/**
 * Defines an interface for all isotonic calibration models.
 */
class MLRLCOMMON_API IIsotonicProbabilityCalibrationModel {
    public:

        virtual ~IIsotonicProbabilityCalibrationModel() {};

        /**
         * A visitor function for handling individual bins.
         */
        typedef std::function<void(uint32 listIndex, float64 threshold, float64 probability)> BinVisitor;

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
 * Defines an interface for all models for the calibration of marginal probabilities via isotonic regression.
 */
class MLRLCOMMON_API IIsotonicMarginalProbabilityCalibrationModel : public IIsotonicProbabilityCalibrationModel,
                                                                    public IMarginalProbabilityCalibrationModel {
    public:

        virtual ~IIsotonicMarginalProbabilityCalibrationModel() override {};
};

/**
 * A model for the calibration of marginal probabilities via isotonic regression.
 */
class IsotonicMarginalProbabilityCalibrationModel final : public IIsotonicMarginalProbabilityCalibrationModel {
    private:

        ListOfLists<Tuple<float64>> binsPerLabel_;

    public:

        /**
         * @param numLabels The total number of available labels
         */
        IsotonicMarginalProbabilityCalibrationModel(uint32 numLabels);

        /**
         * Provides access to the bins that correspond to a specific label and allows to modify them.
         */
        typedef ListOfLists<Tuple<float64>>::row bin_list;

        /**
         * Provides access to the bins that belong to the list at a specific index and allows to modify its elements.
         *
         * @param listIndex The index of the list
         * @return          A `bin_list`
         */
        bin_list operator[](uint32 listIndex);

        float64 calibrateMarginalProbability(uint32 labelIndex, float64 marginalProbability) const override;

        uint32 getNumBinLists() const override;

        void addBin(uint32 listIndex, float64 threshold, float64 probability) override;

        void visit(BinVisitor visitor) const override;
};

/**
 * Creates and returns a new object of the type `IIsotonicMarginalProbabilityCalibrationModel`.
 *
 * @param numLabels The total number of available labels
 * @return          An unique pointer to an object of type `IIsotonicMarginalProbabilityCalibrationModel` that has been
 *                  created
 */
MLRLCOMMON_API std::unique_ptr<IIsotonicMarginalProbabilityCalibrationModel>
  createIsotonicMarginalProbabilityCalibrationModel(uint32 numLabels);

/**
 * Defines an interface for all model for the calibration of joint probabilities via isotonic regression.
 */
class MLRLCOMMON_API IIsotonicJointProbabilityCalibrationModel : public IIsotonicProbabilityCalibrationModel,
                                                                 public IJointProbabilityCalibrationModel {
    public:

        virtual ~IIsotonicJointProbabilityCalibrationModel() override {};
};

/**
 * A model for the calibration of joint probabilities via isotonic regression.
 */
class IsotonicJointProbabilityCalibrationModel final : public IIsotonicJointProbabilityCalibrationModel {
    private:

        ListOfLists<Tuple<float64>> binsPerLabelVector_;

    public:

        /**
         * @param numLabelVectors The total number of known label vectors
         */
        IsotonicJointProbabilityCalibrationModel(uint32 numLabelVectors);

        /**
         * Provides access to the bins that correspond to a specific label vector and allows to modify them.
         */
        typedef ListOfLists<Tuple<float64>>::row bin_list;

        /**
         * Provides access to the bins that belong to the list at a specific index and allows to modify its elements.
         *
         * @param listIndex The index of the list
         * @return          A `bin_list`
         */
        bin_list operator[](uint32 listIndex);

        float64 calibrateJointProbability(uint32 labelVectorIndex, float64 jointProbability) const override;

        uint32 getNumBinLists() const override;

        void addBin(uint32 listIndex, float64 threshold, float64 probability) override;

        void visit(BinVisitor visitor) const override;
};

/**
 * Creates and returns a new object of the type `IIsotonicJointProbabilityCalibrationModel`.
 *
 * @param numLabelVectors   The total number of known label vectors
 * @return                  An unique pointer to an object of type `IIsotonicJointProbabilityCalibrationModel` that has
 *                          been created
 */
MLRLCOMMON_API std::unique_ptr<IIsotonicJointProbabilityCalibrationModel>
  createIsotonicJointProbabilityCalibrationModel(uint32 numLabelVectors);
