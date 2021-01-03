/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/model/model_builder.h"
#include "../../../common/cpp/model/rule.h"
#include <list>


/**
 * A model that stores several rules in a list.
 */
class RuleList final : public IModel {

    private:

        std::list<std::unique_ptr<Rule>> list_;

    public:

        /**
         * Adds a new rule to the model.
         *
         * @param rulePtr An unique pointer to an object of type `Rule` that should be added
         */
        void append(std::unique_ptr<Rule> rulePtr);

        void predict(const CContiguousFeatureMatrix& featureMatrix,
                     DenseMatrix<float64>& predictionMatrix) const override;

        void predict(const CsrFeatureMatrix& featureMatrix, DenseMatrix<float64>& predictionMatrix) const override;

};

/**
 * Allows to build models that store several rules in a list.
 */
class RuleListBuilder final : public IModelBuilder {

    private:

        std::unique_ptr<RuleList> modelPtr_;

    public:

        void setDefaultRule(const AbstractPrediction* prediction) override;

        void addRule(const ConditionList& conditions, const AbstractPrediction& prediction) override;

        std::unique_ptr<IModel> build() override;

};
