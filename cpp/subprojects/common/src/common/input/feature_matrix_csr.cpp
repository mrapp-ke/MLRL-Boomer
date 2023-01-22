#include "common/input/feature_matrix_csr.hpp"
#include "common/output/predictor_classification.hpp"
#include "common/output/predictor_regression.hpp"
#include "common/output/predictor_probability.hpp"
#include "common/prediction/prediction_matrix_dense.hpp"
#include "common/prediction/prediction_matrix_sparse_binary.hpp"
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

// TODO Remove
std::unique_ptr<DensePredictionMatrix<uint8>> CsrFeatureMatrix::predictLabels(
        const IClassificationPredictor& predictor, uint32 numLabels) const {
    return predictor.predict(*this, numLabels);
}

std::unique_ptr<ISparseLabelPredictor> CsrFeatureMatrix::createSparseLabelPredictor(
        const ISparseLabelPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
        uint32 numLabels) const {
    // TODO
    return nullptr;
}

// TODO Remove
std::unique_ptr<BinarySparsePredictionMatrix> CsrFeatureMatrix::predictSparseLabels(
        const IClassificationPredictor& predictor, uint32 numLabels) const {
    return predictor.predictSparse(*this, numLabels);
}

std::unique_ptr<IScorePredictor> CsrFeatureMatrix::createScorePredictor(const IScorePredictorFactory& factory,
                                                                        const IRuleModel& ruleModel,
                                                                        const ILabelSpaceInfo& labelSpaceInfo,
                                                                        uint32 numLabels) const {
    // TODO
    return nullptr;
}

// TODO Remove
std::unique_ptr<DensePredictionMatrix<float64>> CsrFeatureMatrix::predictScores(
        const IOldRegressionPredictor& predictor, uint32 numLabels) const {
    return predictor.predict(*this, numLabels);
}

std::unique_ptr<IProbabilityPredictor> CsrFeatureMatrix::createProbabilityPredictor(
        const IProbabilityPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
        uint32 numLabels) const {
    // TODO
    return nullptr;
}

// TODO Remove
std::unique_ptr<DensePredictionMatrix<float64>> CsrFeatureMatrix::predictProbabilities(
        const IOldProbabilityPredictor& predictor, uint32 numLabels) const {
    return predictor.predict(*this, numLabels);
}

std::unique_ptr<ICsrFeatureMatrix> createCsrFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data,
                                                          uint32* rowIndices, uint32* colIndices) {
    return std::make_unique<CsrFeatureMatrix>(numRows, numCols, data, rowIndices, colIndices);
}
