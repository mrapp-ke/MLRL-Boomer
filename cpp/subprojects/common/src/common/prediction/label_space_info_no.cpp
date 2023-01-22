#include "common/prediction/label_space_info_no.hpp"
#include "common/input/feature_matrix_c_contiguous.hpp"
#include "common/input/feature_matrix_csr.hpp"
#include "common/output/predictor_classification.hpp"
#include "common/output/predictor_regression.hpp"
#include "common/output/predictor_probability.hpp"
#include "common/model/rule_list.hpp"
#include "common/prediction/predictor_label.hpp"
#include "common/prediction/predictor_probability.hpp"
#include "common/prediction/predictor_score.hpp"


/**
 * An implementation of the type `INoLabelSpaceInfo` that does not provide any information about the label space.
 */
class NoLabelSpaceInfo final : public INoLabelSpaceInfo {

    public:

        std::unique_ptr<ILabelPredictor> createLabelPredictor(const ILabelPredictorFactory& factory,
                                                              const CContiguousFeatureMatrix& featureMatrix,
                                                              const RuleList& model, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, numLabels);
        }

        std::unique_ptr<ILabelPredictor> createLabelPredictor(const ILabelPredictorFactory& factory,
                                                              const CsrFeatureMatrix& featureMatrix,
                                                              const RuleList& model, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, numLabels);
        }

        std::unique_ptr<ISparseLabelPredictor> createSparseLabelPredictor(const ISparseLabelPredictorFactory& factory,
                                                                          const CContiguousFeatureMatrix& featureMatrix,
                                                                          const RuleList& model,
                                                                          uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, numLabels);
        }

        std::unique_ptr<ISparseLabelPredictor> createSparseLabelPredictor(const ISparseLabelPredictorFactory& factory,
                                                                          const CsrFeatureMatrix& featureMatrix,
                                                                          const RuleList& model,
                                                                          uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, numLabels);
        }

        // TODO Remove
        std::unique_ptr<IClassificationPredictor> createClassificationPredictor(
                const IClassificationPredictorFactory& factory, const RuleList& model) const override {
            return factory.create(model, nullptr);
        }

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CContiguousFeatureMatrix& featureMatrix,
                                                              const RuleList& model, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, numLabels);
        }

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CsrFeatureMatrix& featureMatrix,
                                                              const RuleList& model, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, numLabels);
        }

        // TODO Remove
        std::unique_ptr<IOldRegressionPredictor> createRegressionPredictor(
                const IRegressionPredictorFactory& factory, const RuleList& model) const override {
            return factory.create(model, nullptr);
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(const IProbabilityPredictorFactory& factory,
                                                                          const CContiguousFeatureMatrix& featureMatrix,
                                                                          const RuleList& model,
                                                                          uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, numLabels);
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(const IProbabilityPredictorFactory& factory,
                                                                          const CsrFeatureMatrix& featureMatrix,
                                                                          const RuleList& model,
                                                                          uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, numLabels);
        }

        // TODO Remove
        std::unique_ptr<IOldProbabilityPredictor> createProbabilityPredictor(
                const IOldProbabilityPredictorFactory& factory, const RuleList& model) const override {
            return factory.create(model, nullptr);
        }

};

std::unique_ptr<INoLabelSpaceInfo> createNoLabelSpaceInfo() {
    return std::make_unique<NoLabelSpaceInfo>();
}
