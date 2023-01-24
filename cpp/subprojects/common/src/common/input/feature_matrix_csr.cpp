#include "common/input/feature_matrix_csr.hpp"
#include "common/prediction/predictor_label.hpp"
#include "common/prediction/predictor_probability.hpp"
#include "common/prediction/predictor_score.hpp"


CsrFeatureMatrix::CsrFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data, uint32* rowIndices,
                                   uint32* colIndices)
    : CsrConstView<const float32>(numRows, numCols, data, rowIndices, colIndices) {

}

bool CsrFeatureMatrix::isSparse() const {
    return true;
}

std::unique_ptr<ILabelPredictor> CsrFeatureMatrix::createLabelPredictor(const ILabelPredictorFactory& factory,
                                                                        const IRuleModel& ruleModel,
                                                                        const ILabelSpaceInfo& labelSpaceInfo,
                                                                        uint32 numLabels) const {
    // TODO
    return nullptr;
}

std::unique_ptr<ISparseLabelPredictor> CsrFeatureMatrix::createSparseLabelPredictor(
        const ISparseLabelPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
        uint32 numLabels) const {
    // TODO
    return nullptr;
}

std::unique_ptr<IScorePredictor> CsrFeatureMatrix::createScorePredictor(const IScorePredictorFactory& factory,
                                                                        const IRuleModel& ruleModel,
                                                                        const ILabelSpaceInfo& labelSpaceInfo,
                                                                        uint32 numLabels) const {
    // TODO
    return nullptr;
}

std::unique_ptr<IProbabilityPredictor> CsrFeatureMatrix::createProbabilityPredictor(
        const IProbabilityPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
        uint32 numLabels) const {
    // TODO
    return nullptr;
}

std::unique_ptr<ICsrFeatureMatrix> createCsrFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data,
                                                          uint32* rowIndices, uint32* colIndices) {
    return std::make_unique<CsrFeatureMatrix>(numRows, numCols, data, rowIndices, colIndices);
}
