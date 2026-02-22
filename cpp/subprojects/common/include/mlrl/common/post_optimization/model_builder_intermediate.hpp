/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/model/model_builder.hpp"

#include <memory>
#include <utility>
#include <vector>

/**
 * An implementation of the class `IModelBuilder` that stores intermediate representations of rules, which can still be
 * modified when globally optimizing a rule-based model once it has been learned, that are ultimately converted into a
 * final model using another `IModelBuilder`.
 */
class IntermediateModelBuilder final : public IModelBuilder {
    public:

        /**
         * The type of a rule, which can still be modified.
         */
        using IntermediateRule = std::pair<std::unique_ptr<ConditionList>, std::unique_ptr<IEvaluatedPrediction>>;

    private:

        const std::unique_ptr<IModelBuilder> modelBuilderPtr_;

        std::unique_ptr<IEvaluatedPrediction> defaultPredictionPtr_;

        std::vector<IntermediateRule> intermediateRuleList_;

        uint32 numUsedRules_;

    public:

        /**
         * @param modelBuilderPtr An unique pointer to an object of type `IModelBuilder` that should be used to build
         *                        the final model
         */
        IntermediateModelBuilder(std::unique_ptr<IModelBuilder> modelBuilderPtr);

        /**
         * An iterator that provides access to the intermediate representations of rules and allows to modify them.
         */
        using iterator = std::vector<IntermediateRule>::iterator;

        /**
         * Returns an `iterator` to the beginning of the intermediate representations of rules.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the intermediate representations of rules.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Removes the intermediate representation of the last rule.
         */
        void removeLastRule();

        /**
         * Returns the total number of rules.
         *
         * @return The total number of rules
         */
        uint32 getNumRules() const;

        /**
         * Returns the number of used rules.
         *
         * @return The number of used rules
         */
        uint32 getNumUsedRules() const;

        void setNumUsedRules(uint32 numUsedRules) override;

        void setDefaultRule(std::unique_ptr<IEvaluatedPrediction>& predictionPtr) override;

        void addRule(std::unique_ptr<ConditionList>& conditionListPtr,
                     std::unique_ptr<IEvaluatedPrediction>& predictionPtr) override;

        std::unique_ptr<IRuleModel> buildModel() override;
};
