#include "mlrl/boosting/util/blas.hpp"

namespace boosting {

    template<typename T>
    Blas<T>::Blas(const Routines& routines) : dot_(routines.dot), spmv_(routines.spmv) {}

    template<typename T>
    T Blas<T>::dot(T* x, T* y, int n) const {
        // Storage spacing between the elements of the arrays x and y
        int inc = 1;
        // Invoke the DOT routine...
        return dot_(&n, x, &inc, y, &inc);
    }

    template<typename T>
    void Blas<T>::spmv(T* a, T* x, T* output, int n) const {
        // "U" if the upper-right triangle of A should be used, "L" if the lower-left triangle should be used
        char* uplo = const_cast<char*>("U");
        // A scalar to be multiplied with the matrix A
        T alpha = 1;
        // The increment for the elements of x and y
        int inc = 1;
        // A scalar to be multiplied with vector y
        T beta = 0;
        // Invoke the SPMV routine...
        spmv_(uplo, &n, &alpha, a, x, &inc, &beta, output, &inc);
    }

    template class Blas<float32>;
    template class Blas<float64>;

    BlasFactory::BlasFactory(const Blas<float32>::Routines& float32Routines,
                             const Blas<float64>::Routines& float64Routines)
        : float32Routines_(float32Routines), float64Routines_(float64Routines) {}

    std::unique_ptr<Blas<float32>> BlasFactory::create32Bit() const {
        return std::make_unique<Blas<float32>>(float32Routines_);
    }

    std::unique_ptr<Blas<float64>> BlasFactory::create64Bit() const {
        return std::make_unique<Blas<float64>>(float64Routines_);
    }

}
