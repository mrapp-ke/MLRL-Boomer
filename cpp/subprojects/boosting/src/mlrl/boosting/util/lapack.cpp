#include "mlrl/boosting/util/lapack.hpp"

#include <stdexcept>
#include <string>

namespace boosting {

    template<typename T>
    Lapack<T>::Lapack(const Routines& routines) : sysv_(routines.sysv) {}

    template<typename T>
    int Lapack<T>::querySysvLworkParameter(T* tmpArray1, T* output, int n) const {
        // "U" if the upper-right triangle of A should be used, "L" if the lower-left triangle should be used
        char* uplo = const_cast<char*>("U");
        // The number of right-hand sides, i.e, the number of columns of the matrix B
        int nrhs = 1;
        // Set "lwork" parameter to -1, which indicates that the optimal value should be queried
        int lwork = -1;
        // Variable to hold the queried value
        T worksize;
        // Variable to hold the result of the solver. Will be 0 when terminated successfully, unlike 0 otherwise
        int info;

        // Query the optimal value for the "lwork" parameter...
        sysv_(uplo, &n, &nrhs, tmpArray1, &n, reinterpret_cast<int*>(0), output, &n, &worksize, &lwork, &info);

        if (info != 0) {
            throw std::runtime_error(
              "SYSV terminated with non-zero info code when querying the optimal lwork parameter: "
              + std::to_string(info));
        }

        return static_cast<int>(worksize);
    }

    template<typename T>
    void Lapack<T>::sysv(T* tmpArray1, int* tmpArray2, T* tmpArray3, T* output, int n, int lwork) const {
        // "U" if the upper-right triangle of A should be used, "L" if the lower-left triangle should be used
        char* uplo = const_cast<char*>("U");
        // The number of right-hand sides, i.e, the number of columns of the matrix B
        int nrhs = 1;
        // Variable to hold the result of the solver. Will be 0 when terminated successfully, unlike 0 otherwise
        int info;

        // Run the SYSV solver...
        sysv_(uplo, &n, &nrhs, tmpArray1, &n, tmpArray2, output, &n, tmpArray3, &lwork, &info);

        if (info != 0) {
            throw std::runtime_error("SYSV terminated with non-zero info code: " + std::to_string(info));
        }
    }

    template class Lapack<float32>;
    template class Lapack<float64>;

    LapackFactory::LapackFactory(const Lapack<float32>::Routines& float32Routines,
                                 const Lapack<float64>::Routines& float64Routines)
        : float32Routines_(float32Routines), float64Routines_(float64Routines) {}

    std::unique_ptr<Lapack<float32>> LapackFactory::create32Bit() const {
        return std::make_unique<Lapack<float32>>(float32Routines_);
    }

    std::unique_ptr<Lapack<float64>> LapackFactory::create64Bit() const {
        return std::make_unique<Lapack<float64>>(float64Routines_);
    }

}
