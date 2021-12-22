#include "boosting/losses/loss_label_wise_squared_error.hpp"


namespace boosting {

    static inline void updateGradientAndHessian(bool trueLabel, float64 predictedScore, float64* gradient,
                                                float64* hessian) {
        float64 expectedScore = trueLabel ? 1 : -1;
        *gradient = (predictedScore - expectedScore);
        *hessian = 1;
    }

    static inline float64 evaluatePrediction(bool trueLabel, float64 predictedScore) {
        float64 expectedScore = trueLabel ? 1 : -1;
        float64 difference = (expectedScore - predictedScore);
        return difference * difference;
    }

    /**
     * An implementation of the type `ILabelWiseLoss` that implements a multi-label variant of the squared error loss
     * that is applied label-wise.
     */
    class LabelWiseSquaredErrorLoss final : public AbstractLabelWiseLoss {

        public:

            LabelWiseSquaredErrorLoss()
                : AbstractLabelWiseLoss(&updateGradientAndHessian, &evaluatePrediction) {

            }

    };

    std::unique_ptr<ILabelWiseLoss> LabelWiseSquaredErrorLossFactory::createLabelWiseLoss() const {
        return std::make_unique<LabelWiseSquaredErrorLoss>();
    }

}
