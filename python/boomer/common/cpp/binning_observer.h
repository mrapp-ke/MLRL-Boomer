#include "arrays.h"
#include "abstract_binning.h"

class BinningObserver{

    public:

        void onBinUpdate(intp binIndex, IndexedFloat32* updatedMatrix);

        BinningObserver(AbstractBinning* toWatch);

}