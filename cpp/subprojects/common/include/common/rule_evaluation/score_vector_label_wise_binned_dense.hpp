/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "score_vector_label_wise.hpp"
#include "score_vector_binned_dense.hpp"
#include "../data/vector_dense.hpp"
#include <iterator>


/**
 * An one-dimensional vector that stores the scores that may be predicted by a rule, corresponding to bins for which the
 * same prediction is made, as well as corresponding quality scores that assess the quality of individual scores, in
 * C-contiguous arrays.
 *
 * @tparam T The type of the vector that provides access to the indices of the labels for which the rule may predict
 */
template<class T>
class DenseBinnedLabelWiseScoreVector final : public DenseBinnedScoreVector<T>, virtual public ILabelWiseScoreVector {

    private:

        DenseVector<float64> qualityScoreVector_;

    public:

        /**
         * Allows to iterate the quality scores that correspond to individual labels.
         */
        class QualityScoreIterator final {

            private:

                const DenseBinnedLabelWiseScoreVector<T>& vector_;

                uint32 index_;

            public:

                QualityScoreIterator(const DenseBinnedLabelWiseScoreVector<T>& vector, uint32 index);

                typedef int difference_type;

                typedef float64 value_type;

                typedef float64* pointer;

                typedef float64 reference;

                typedef std::random_access_iterator_tag iterator_category;

                reference operator[](uint32 index) const;

                reference operator*() const;

                QualityScoreIterator& operator++();

                QualityScoreIterator& operator++(int n);

                QualityScoreIterator& operator--();

                QualityScoreIterator& operator--(int n);

                bool operator!=(const QualityScoreIterator& rhs) const;

                difference_type operator-(const QualityScoreIterator& rhs) const;

        };

        /**
         * @param labelIndices  A reference to an object of template type `T` that provides access to the indices of
         *                      the labels for which the rule may predict
         * @param numBins       The number of bins
         */
        DenseBinnedLabelWiseScoreVector(const T& labelIndices, uint32 numBins);

        typedef QualityScoreIterator quality_score_const_iterator;

        typedef DenseVector<float64>::iterator quality_score_binned_iterator;

        typedef DenseVector<float64>::const_iterator quality_score_binned_const_iterator;

        /**
         * Returns a `quality_score_const_iterator` to the beginning of the quality scores that correspond to the
         * labels.
         *
         * @return A `quality_score_const_iterator` to the beginning
         */
        quality_score_const_iterator quality_scores_cbegin() const;

        /**
         * Returns a `quality_score_const_iterator` to the end of the quality scores that correspond to the labels.
         *
         * @return A `quality_score_const_iterator` to the end
         */
        quality_score_const_iterator quality_scores_cend() const;

        /**
         * Returns a `quality_score_binned_iterator` to the beginning of the quality scores that correspond to the bins.
         *
         * @return A `quality_score_binned_iterator` to the beginning
         */
        quality_score_binned_iterator quality_scores_binned_begin();

        /**
         * Returns a `quality_score_binned_iterator` to the end of the quality scores that correspond to the bins.
         *
         * @return A `quality_score_binned_iterator` to the end
         */
        quality_score_binned_iterator quality_scores_binned_end();

        /**
         * Returns a `quality_score_binned_const_iterator` to the beginning of the quality scores that correspond to the bins.
         *
         * @return A `quality_score_binned_const_iterator` to the beginning
         */
        quality_score_binned_const_iterator quality_scores_binned_cbegin() const;

        /**
         * Returns a `quality_score_binned_const_iterator` to the end of the quality scores that correspond to the bins.
         *
         * @return A `quality_score_binned_const_iterator` to the end
         */
        quality_score_binned_const_iterator quality_scores_binned_cend() const;

        const AbstractEvaluatedPrediction* processScores(const AbstractEvaluatedPrediction* bestHead,
                                                         ILabelWiseScoreProcessor& scoreProcessor) const override;

};
