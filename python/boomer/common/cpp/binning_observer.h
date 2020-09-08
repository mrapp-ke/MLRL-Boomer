#include "arrays.h"
#include "abstract_binning.h"

class BinningObserver{

    public:

        virtual void onBinUpdate(intp binIndex, IndexedFloat32* indexedValue);

};