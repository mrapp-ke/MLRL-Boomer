#include "predictor_classification_example_wise.h"
#include "predictor_common.h"
#include "../../../common/cpp/data/matrix_dense.h"
#include <algorithm>
#include <iostream>


namespace boosting {

    template<class T>
    static inline void printKnownLabelVectors(const T& labelVectors, uint32 numLabels) {
        std::cout << "Known label vectors\n";
        std::cout << "===================\n";

        uint32 no = 0;

        for (auto it = labelVectors.cbegin(); it != labelVectors.cend(); it++) {
            std::cout << no << ": ";
            const std::unique_ptr<LabelVector>& labelVectorPtr = *it;
            auto indexIterator = labelVectorPtr->indices_cbegin();
            auto indicesEnd = labelVectorPtr->indices_cend();

            for (uint32 i = 0; i < numLabels; i++) {
                bool label;

                if (indexIterator != indicesEnd && *indexIterator == i) {
                    indexIterator++;
                    label = true;
                } else {
                    label = false;
                }

                std::cout << (label ? "1" : "0") << " ";
            }

            std::cout << "\n";
            no++;
        }

        std::cout << "\n";
    }

    template<class T>
    static inline void predictClosestLabelVector(uint32 exampleIndex, const DenseMatrix<float64>& scoreMatrix,
                                                 CContiguousView<uint8>& predictionMatrix, const IMeasure& measure,
                                                 const T& labelVectors) {
        std::cout << "\n====================================================================\n";
        std::cout << "Score vector\n";
        std::cout << "------------\n";

        for (auto it = scoreMatrix.row_cbegin(exampleIndex); it != scoreMatrix.row_cend(exampleIndex); it++) {
            float64 score = *it;
            std::cout << score << " ";
        }

        std::cout << "\n\n";


        std::fill(predictionMatrix.row_begin(exampleIndex), predictionMatrix.row_end(exampleIndex), 0);
        const LabelVector* closestLabelVector = nullptr;
        float64 bestScore = 0;


        std::cout << "Comparisons\n";
        std::cout << "-----------\n";

        uint32 no = 0;

        for (auto it = labelVectors.cbegin(); it != labelVectors.cend(); it++) {
            std::cout << no << ": ";
            const std::unique_ptr<LabelVector>& labelVectorPtr = *it;
            float64 score = measure.evaluate(exampleIndex, *labelVectorPtr, scoreMatrix);
            std::cout << "score = " << score;

            if (closestLabelVector == nullptr || score < bestScore) {
                closestLabelVector = labelVectorPtr.get();
                bestScore = score;
                std::cout << " --> best seen so far";
            }

            std::cout << "\n";
            no++;
        }

        std::cout << "\n";
        std::cout << "Prediction\n";
        std::cout << "----------\n";

        if (closestLabelVector != nullptr) {
            CContiguousView<uint8>::iterator iterator = predictionMatrix.row_begin(exampleIndex);

            for (auto it = closestLabelVector->indices_cbegin(); it != closestLabelVector->indices_cend(); it++) {
                uint32 labelIndex = *it;
                iterator[labelIndex] = 1;
            }

            for (auto it = predictionMatrix.row_cbegin(exampleIndex); it != predictionMatrix.row_cend(exampleIndex); it++) {
                std::cout << *it << " ";
            }

            std::cout << "\n";
        } else {
            std::cout << "no prediction possible!?\n";
        }
    }

    template<class T>
    static inline void predictInternally(const RuleModel& model, const CContiguousFeatureMatrix& featureMatrix,
                                         DenseMatrix<float64>& scoreMatrix, CContiguousView<uint8>& predictionMatrix,
                                         const IMeasure& measure, const T& labelVectors) {
        uint32 numExamples = featureMatrix.getNumRows();

        for (uint32 i = 0; i < numExamples; i++) {
            for (auto it = model.cbegin(); it != model.cend(); it++) {
                const Rule& rule = *it;
                applyRule(rule, featureMatrix.row_cbegin(i), featureMatrix.row_cend(i), scoreMatrix.row_begin(i));
            }

            predictClosestLabelVector(i, scoreMatrix, predictionMatrix, measure, labelVectors);
        }
    }

    template<class T>
    static inline void predictInternally(const RuleModel& model, const CsrFeatureMatrix& featureMatrix,
                                         DenseMatrix<float64>& scoreMatrix, CContiguousView<uint8>& predictionMatrix,
                                         const IMeasure& measure, const T& labelVectors) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        float32 tmpArray1[numFeatures];
        uint32 tmpArray2[numFeatures] = {};
        uint32 n = 1;

        for (uint32 i = 0; i < numExamples; i++) {
            for (auto it = model.cbegin(); it != model.cend(); it++) {
                const Rule& rule = *it;
                applyRuleCsr(rule, featureMatrix.row_indices_cbegin(i), featureMatrix.row_indices_cend(i),
                             featureMatrix.row_values_cbegin(i), featureMatrix.row_values_cend(i),
                             scoreMatrix.row_begin(i), &tmpArray1[0], &tmpArray2[0], n);
                n++;
            }

            predictClosestLabelVector(i, scoreMatrix, predictionMatrix, measure, labelVectors);
        }
    }

    ExampleWiseClassificationPredictor::ExampleWiseClassificationPredictor(std::shared_ptr<IMeasure> measurePtr)
        : measurePtr_(measurePtr) {

    }

    void ExampleWiseClassificationPredictor::addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr) {
        labelVectors_.emplace(std::move(labelVectorPtr));
    }

    void ExampleWiseClassificationPredictor::visit(LabelVectorVisitor visitor) const {
        for (auto it = labelVectors_.cbegin(); it != labelVectors_.cend(); it++) {
            const std::unique_ptr<LabelVector>& labelVectorPtr = *it;
            visitor(*labelVectorPtr);
        }
    }

    void ExampleWiseClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                                     CContiguousView<uint8>& predictionMatrix,
                                                     const RuleModel& model) const {
                                                     uint32 numExamples = predictionMatrix.getNumRows();
        std::cout << "Predicting based on CContiguousFeatureMatrix...\n";
        uint32 numLabels = predictionMatrix.getNumCols();
        printKnownLabelVectors(labelVectors_, numLabels);
        DenseMatrix<float64> scoreMatrix(numExamples, numLabels);
        predictInternally<LabelVectorSet>(model, featureMatrix, scoreMatrix, predictionMatrix, *measurePtr_,
                                          labelVectors_);
    }

    void ExampleWiseClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                                     CContiguousView<uint8>& predictionMatrix,
                                                     const RuleModel& model) const {
        std::cout << "Predicting based on CsrFeatureMatrix...\n";
        uint32 numExamples = predictionMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        printKnownLabelVectors(labelVectors_, numLabels);
        DenseMatrix<float64> scoreMatrix(numExamples, numLabels);
        predictInternally<LabelVectorSet>(model, featureMatrix, scoreMatrix, predictionMatrix, *measurePtr_,
                                          labelVectors_);
    }

}
