/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../input/nominal_feature_mask.h"
#include "../input/feature_matrix.h"
#include "../input/label_matrix.h"
#include "../model/model_builder.h"


/**
 * Defines an interface for all classes that implement an algorithm for inducing several rules that will be added to a
 * resulting `RuleModel`.
 */
class IRuleModelInduction {

    public:

        virtual ~IRuleModelInduction() { };

        /**
         * Trains and returns a `RuleModel` that consists of several rules.
         *
         * @param nominalFeatureMask    A reference to an object of type `INominalFeatureMask` that provides access to
         *                              the information whether individual features are nominal or not
         * @param featureMatrix         A reference to an object of type `IFeatureMatrix` that provides access to the
         *                              feature values of the training examples
         * @param labelMatrix           A reference to an object of type `ILabelMatrix` that provides access to the
         *                              labels of the training examples
         * @param modelBuilder          A reference to an object of type `IModelBuilder`, the induced rules should be
         *                              added to
         * @return                      An unique pointer to an object of type `RuleModel` that consists of the rules
         *                              that have been induced
         */
        virtual std::unique_ptr<RuleModel> induceRules(const INominalFeatureMask& nominalFeatureMask,
                                                       const IFeatureMatrix& featureMatrix,
                                                       const ILabelMatrix& labelMatrix,
                                                       IModelBuilder& modelBuilder) const = 0;

};
