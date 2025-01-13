/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

namespace boosting {

    /**
     * An individual label space statistic that consists of a gradient and a Hessian.
     *
     * @tparam T The type of the gradient and Hessian
     */
    template<typename T>
    struct Statistic final {
        public:

            Statistic() {}

            /**
             * @param gradient  The gradient
             * @param hessian   The Hessian
             */
            Statistic(T gradient, T hessian) : gradient(gradient), hessian(hessian) {}

            /**
             * The gradient.
             */
            T gradient;

            /**
             * The Hessian.
             */
            T hessian;

            /**
             * Assigns a specific value to the gradient and Hessian of this statistic.
             *
             * @param rhs   A reference to the value to be assigned
             * @return      A reference to the modified statistic
             */
            Statistic<T>& operator=(const T& rhs) {
                gradient = rhs;
                hessian = rhs;
                return *this;
            }

            /**
             * Adds a specific value to the gradient and Hessian of this statistic.
             *
             * @param rhs   A reference to the value to be added
             * @return      A reference to the modified statistic
             */
            Statistic<T>& operator+=(const T& rhs) {
                gradient += rhs;
                hessian += rhs;
                return *this;
            }

            /**
             * Creates and returns a new statistic that results from adding a specific value to the gradient and Hessian
             * of an existing statistic.
             *
             * @param lhs   The original statistic
             * @param rhs   A reference to the value to be added
             * @return      The statistic that has been created
             */
            friend Statistic<T> operator+(Statistic<T> lhs, const T& rhs) {
                lhs += rhs;
                return lhs;
            }

            /**
             * Adds the gradient and Hessian of a given statistic to the gradient and Hessian of this statistic,
             *
             * @param rhs   A reference to the statistic, whose gradient and Hessian should be added
             * @return      A reference to the modified statistic
             */
            Statistic<T>& operator+=(const Statistic<T>& rhs) {
                gradient += rhs.gradient;
                hessian += rhs.hessian;
                return *this;
            }

            /**
             * Creates and returns a new statistic that results from adding the gradient and Hessian of a specific
             * statistic to the gradient and Hessian of an existing statistic.
             *
             * @param lhs   The original statistic
             * @param rhs   A reference to the statistic, whose gradient and Hessian should be added
             * @return      The statistic that has been created
             */
            friend Statistic<T> operator+(Statistic<T> lhs, const Statistic<T>& rhs) {
                lhs += rhs;
                return lhs;
            }

            /**
             * Subtracts a specific value from the gradient and Hessian of this statistic.
             *
             * @param rhs   A reference to the value to be subtracted
             * @return      A reference to the modified statistic
             */
            Statistic<T>& operator-=(const T& rhs) {
                gradient -= rhs;
                hessian -= rhs;
                return *this;
            }

            /**
             * Creates and returns a new statistic that results from subtracting a specific value from the gradient and
             * Hessian of an existing statistic.
             *
             * @param lhs   The original statistic
             * @param rhs   A reference to the value to be subtracted
             * @return      The statistic that has been created
             */
            friend Statistic<T> operator-(Statistic<T> lhs, const T& rhs) {
                lhs -= rhs;
                return lhs;
            }

            /**
             * Subtracts the gradient and Hessian of a given statistic from the gradient and Hessian of this statistic.
             *
             * @param rhs   A reference to the statistic, whose gradient and Hessian should be subtracted
             * @return      A reference to the modified statistic
             */
            Statistic<T>& operator-=(const Statistic<T>& rhs) {
                gradient -= rhs.gradient;
                hessian -= rhs.hessian;
                return *this;
            }

            /**
             * Creates and returns a new statistic that results from subtracting the gradient and Hessian of a specific
             * statistic from the gradient and Hessian of an existing statistic.
             *
             * @param lhs   The original statistic
             * @param rhs   A reference to the value to be subtracted
             * @return      The statistic that has been created
             */
            friend Statistic<T> operator-(Statistic<T> lhs, const Statistic<T>& rhs) {
                lhs -= rhs;
                return lhs;
            }

            /**
             * Multiplies the gradient and Hessian of this statistic with a specific value.
             *
             * @param rhs   A reference to the value to be multiplied by
             * @return      A reference to the modified statistic
             */
            Statistic<T>& operator*=(const T& rhs) {
                gradient *= rhs;
                hessian *= rhs;
                return *this;
            }

            /**
             * Creates and returns a new statistic that results from multiplying the gradient and Hessian of an existing
             * statistic with a specific value.
             *
             * @param lhs   The original statistic
             * @param rhs   A reference to the value to be multiplied by
             * @return      The statistic that has been created
             */
            friend Statistic<T> operator*(Statistic<T> lhs, const T& rhs) {
                lhs *= rhs;
                return lhs;
            }
    };

}
