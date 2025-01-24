#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/rule_refinement/prediction.hpp"
#include "mlrl/common/rule_refinement/score_processor.hpp"

template<typename ScoreType, typename IndexVector>
DenseScoreVector<ScoreType, IndexVector>::DenseScoreVector(const IndexVector& outputIndices, bool sorted)
    : ViewDecorator<AllocatedView<ScoreType>>(AllocatedView<ScoreType>(outputIndices.getNumElements())),
      outputIndices_(outputIndices), sorted_(sorted) {}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::index_const_iterator
  DenseScoreVector<ScoreType, IndexVector>::indices_cbegin() const {
    return outputIndices_.cbegin();
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::index_const_iterator
  DenseScoreVector<ScoreType, IndexVector>::indices_cend() const {
    return outputIndices_.cend();
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::value_iterator
  DenseScoreVector<ScoreType, IndexVector>::values_begin() {
    return this->view.begin();
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::value_iterator
  DenseScoreVector<ScoreType, IndexVector>::values_end() {
    return &this->view.array[this->getNumElements()];
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::value_const_iterator
  DenseScoreVector<ScoreType, IndexVector>::values_cbegin() const {
    return this->view.cbegin();
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::value_const_iterator
  DenseScoreVector<ScoreType, IndexVector>::values_cend() const {
    return &this->view.array[this->getNumElements()];
}

template<typename ScoreType, typename IndexVector>
uint32 DenseScoreVector<ScoreType, IndexVector>::getNumElements() const {
    return outputIndices_.getNumElements();
}

template<typename ScoreType, typename IndexVector>
bool DenseScoreVector<ScoreType, IndexVector>::isPartial() const {
    return outputIndices_.isPartial();
}

template<typename ScoreType, typename IndexVector>
bool DenseScoreVector<ScoreType, IndexVector>::isSorted() const {
    return sorted_;
}

template<typename ScoreType, typename IndexVector>
void DenseScoreVector<ScoreType, IndexVector>::updatePrediction(IPrediction& prediction) const {
    prediction.set(this->values_cbegin(), this->values_cend());
}

template<typename ScoreType, typename IndexVector>
void DenseScoreVector<ScoreType, IndexVector>::processScores(ScoreProcessor& scoreProcessor) const {
    scoreProcessor.processScores(*this);
}

template class DenseScoreVector<float32, PartialIndexVector>;
template class DenseScoreVector<float64, PartialIndexVector>;
template class DenseScoreVector<float32, CompleteIndexVector>;
template class DenseScoreVector<float64, CompleteIndexVector>;
