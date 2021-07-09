/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_evaluation/score_vector_label_wise.hpp"
#include "common/rule_evaluation/score_vector_binned_dense.hpp"
#include "common/data/vector_dense.hpp"
#include <iterator>


/**
 * An one-dimensional vector that stores the scores that may be predicted by a rule, corresponding to bins for which the
 * same prediction is made, as well as corresponding quality scores that assess the quality of individual scores, in
 * C-contiguous arrays.
 *
 * @tparam T The type of the vector that provides access to the indices of the labels for which the rule may predict
 */
template<typename T>
class DenseBinnedLabelWiseScoreVector final : public DenseBinnedScoreVector<T>, virtual public ILabelWiseScoreVector {

    private:

        /**
         * An iterator that provides read-only access to the quality scores of all elements in a
         * `DenseBinnedLabelWiseScoreVector`.
         */
        class QualityScoreConstIterator final {

            private:

                DenseVector<uint32>::const_iterator binIndexIterator_;

                DenseVector<float64>::const_iterator qualityScoreIterator_;

            public:

                /**
                 * @param binIndexIterator      An iterator to the bin indices of individual elements
                 * @param qualityScoreIterator  An iterator to the quality scores of individual bins
                 */
                QualityScoreConstIterator(DenseVector<uint32>::const_iterator binIndexIterator,
                                          DenseVector<float64>::const_iterator qualityScoreIterator);

                /**
                 * The type that is used to represent the difference between two iterators.
                 */
                typedef int difference_type;

                /**
                 * The type of the elements, the iterator provides access to.
                 */
                typedef float64 value_type;

                /**
                 * The type of a pointer to an element, the iterator provides access to.
                 */
                typedef const float64* pointer;

                /**
                 * The type of a reference to an element, the iterator provides access to.
                 */
                typedef const float64& reference;

                /**
                 * The tag that specifies the capabilities of the iterator.
                 */
                typedef std::random_access_iterator_tag iterator_category;

                /**
                 * Returns the element at a specific index.
                 *
                 * @param index The index of the element to be returned
                 * @return      The element at the given index
                 */
                reference operator[](uint32 index) const;

                /**
                 * Returns the element, the iterator currently refers to.
                 *
                 * @return The element, the iterator currently refers to
                 */
                reference operator*() const;

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator to the next element
                 */
                QualityScoreConstIterator& operator++();

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator to the next element
                 */
                QualityScoreConstIterator& operator++(int n);

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator to the previous element
                 */
                QualityScoreConstIterator& operator--();

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator to the previous element
                 */
                QualityScoreConstIterator& operator--(int n);

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators do not refer to the same element, false otherwise
                 */
                bool operator!=(const QualityScoreConstIterator& rhs) const;

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators refer to the same element, false otherwise
                 */
                bool operator==(const QualityScoreConstIterator& rhs) const;

                /**
                 * Returns the difference between this iterator and another one.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      The difference between the iterators
                 */
                difference_type operator-(const QualityScoreConstIterator& rhs) const;

        };

        DenseVector<float64> qualityScoreVector_;

    public:

        /**
         * @param labelIndices  A reference to an object of template type `T` that provides access to the indices of
         *                      the labels for which the rule may predict
         * @param numBins       The number of bins
         */
        DenseBinnedLabelWiseScoreVector(const T& labelIndices, uint32 numBins);

        /**
         * An iterator that provides read-only access to the quality scores that correspond to individual labels.
         */
        typedef QualityScoreConstIterator quality_score_const_iterator;

        /**
         * An iterator that provides access to the quality scores that correspond to individual bins and allows to
         * modify them.
         */
        typedef DenseVector<float64>::iterator quality_score_binned_iterator;

        /**
         * An iterator that provides read-only access to the quality scores that correspond to individual bins.
         */
        typedef DenseVector<float64>::const_iterator quality_score_binned_const_iterator;

        /**
         * Returns a `quality_score_const_iterator` to the beginning of the quality scores that correspond to individual
         * labels.
         *
         * @return A `quality_score_const_iterator` to the beginning
         */
        quality_score_const_iterator quality_scores_cbegin() const;

        /**
         * Returns a `quality_score_const_iterator` to the end of the quality scores that correspond to individual
         * labels.
         *
         * @return A `quality_score_const_iterator` to the end
         */
        quality_score_const_iterator quality_scores_cend() const;

        /**
         * Returns a `quality_score_binned_iterator` to the beginning of the quality scores that correspond to
         * individual bins.
         *
         * @return A `quality_score_binned_iterator` to the beginning
         */
        quality_score_binned_iterator quality_scores_binned_begin();

        /**
         * Returns a `quality_score_binned_iterator` to the end of the quality scores that correspond to individual
         * bins.
         *
         * @return A `quality_score_binned_iterator` to the end
         */
        quality_score_binned_iterator quality_scores_binned_end();

        /**
         * Returns a `quality_score_binned_const_iterator` to the beginning of the quality scores that correspond to
         * individual bins.
         *
         * @return A `quality_score_binned_const_iterator` to the beginning
         */
        quality_score_binned_const_iterator quality_scores_binned_cbegin() const;

        /**
         * Returns a `quality_score_binned_const_iterator` to the end of the quality scores that correspond to
         * individual bins.
         *
         * @return A `quality_score_binned_const_iterator` to the end
         */
        quality_score_binned_const_iterator quality_scores_binned_cend() const;

        const AbstractEvaluatedPrediction* processScores(const AbstractEvaluatedPrediction* bestHead,
                                                         ILabelWiseScoreProcessor& scoreProcessor) const override;

};
