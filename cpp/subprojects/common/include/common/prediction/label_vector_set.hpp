/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/arrays.hpp"
#include "common/input/label_matrix_row_wise.hpp"
#include "common/prediction/label_space_info.hpp"

#include <functional>
#include <memory>
#include <unordered_map>

/**
 * Defines an interface for all classes that provide access to a set of unique label vectors.
 */
class MLRLCOMMON_API ILabelVectorSet : public ILabelSpaceInfo {
    public:

        virtual ~ILabelVectorSet() override {};

        /**
         * A visitor function for handling objects of the type `LabelVector`.
         */
        typedef std::function<void(const LabelVector&)> LabelVectorVisitor;

        /**
         * Adds a label vector to the set.
         *
         * @param labelVectorPtr An unique pointer to an object of type `LabelVector`
         */
        virtual void addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr) = 0;

        /**
         * Invokes the given visitor function for each label vector that has been added to the set.
         *
         * @param visitor The visitor function for handling objects of the type `LabelVector`
         */
        virtual void visit(LabelVectorVisitor visitor) const = 0;
};

/**
 * An implementation of the type `ILabelVectorSet` that stores a set of unique label vectors, as well as their
 * frequency.
 */
class LabelVectorSet final : public ILabelVectorSet {
    private:

        /**
         * Allows to compute hashes for objects of type `LabelVector`.
         */
        struct Hash final {
            public:

                inline std::size_t operator()(const std::unique_ptr<LabelVector>& v) const {
                    return hashArray(v->cbegin(), v->getNumElements());
                }
        };

        /**
         * Allows to check whether two objects of type `LabelVector` are equal or not.
         */
        struct Pred final {
            public:

                inline bool operator()(const std::unique_ptr<LabelVector>& lhs,
                                       const std::unique_ptr<LabelVector>& rhs) const {
                    return compareArrays(lhs->cbegin(), lhs->getNumElements(), rhs->cbegin(), rhs->getNumElements());
                }
        };

        typedef std::unordered_map<std::unique_ptr<LabelVector>, uint32, Hash, Pred> Map;

        Map labelVectors_;

    public:

        /**
         * An iterator that provides read-only access to the label vectors, as well as their frequency.
         */
        typedef Map::const_iterator const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the label vectors in the set.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the label vectors in the set.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns the number of label vectors in the set.
         *
         * @return The number of label vectors
         */
        uint32 getNumLabelVectors() const;

        void addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr) override;

        void visit(LabelVectorVisitor visitor) const override;

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
          const RuleList& ruleList, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& ruleList,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
          const RuleList& ruleList, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& ruleList,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CContiguousFeatureMatrix& featureMatrix,
                                                              const RuleList& ruleList,
                                                              uint32 numLabels) const override;

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CsrFeatureMatrix& featureMatrix,
                                                              const RuleList& ruleList,
                                                              uint32 numLabels) const override;

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
          const RuleList& ruleList, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& ruleList,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;
};

/**
 * Creates and returns a new object of the type `ILabelVectorSet`.
 *
 * @return An unique pointer to an object of type `ILabelVectorSet` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ILabelVectorSet> createLabelVectorSet();

/**
 * Creates and returns a new object of the type `ILabelVectorSet` that stores all label vectors that are encountered in
 * a given label matrix.
 *
 * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix`
 * @return              An unique pointer to an object of type `ILabelVectorSet` that has been created
 */
std::unique_ptr<ILabelVectorSet> createLabelVectorSet(const IRowWiseLabelMatrix& labelMatrix);
