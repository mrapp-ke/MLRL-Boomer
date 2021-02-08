/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once


/**
 * Defines an interface for all classes that provide access to the indices of training examples that have been split
 * into a training set and a holdout set.
 */
class IPartition {

    public:

        virtual ~IPartition() { };

};
