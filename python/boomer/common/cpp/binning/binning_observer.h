/**
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/types.h"


/**
 * Defines an interface to be implemented by classes that should be notified when values are assigned to bins.
 *
 * @tparam T The type of the values that are assigned to bins
 */
template<class T>
class IBinningObserver {

    public:

        virtual ~IBinningObserver() { };

        /**
         * Notifies the observer that a value has been assigned to a certain bin.
         *
         * @param binIndex      The index of the bin, the value is assigned to
         * @param originalIndex The original index of the value
         * @param value         The value
         */
        virtual void onBinUpdate(uint32 binIndex, uint32 originalIndex, T value) = 0;

};
