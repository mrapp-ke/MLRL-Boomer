#!/bin/bash

if [[ "$#" -gt 1 ]]; then
    echo "Illegal number of arguments!"
    exit 7
fi

DATASET=$1

# ceiling for PEAK_LABEL
function ceiling {
  CEIL=${1/.*}
  DECIMAL=${1##*.}

  if [[ DECIMAL -gt 0 && DECIMAL -ne CEIL ]]; then # if no decimal point DECIMAL is equal to CEIL
    CEIL=$((CEIL + 1))
  fi

  echo "$CEIL"
}

# cardinality of dataset
case $DATASET in

  bibtex)
    CARDINALITY=2.402
    ;;

  birds)
    CARDINALITY=1.014
    ;;

  cal500)
    CARDINALITY=26.044
    ;;

  corel5k)
    CARDINALITY=3.522
    ;;

  emotions)
    CARDINALITY=1.868
    ;;

  enron)
    CARDINALITY=3.378
    ;;

  eukaryote-pse-aac)
    CARDINALITY=1.146
    ;;

  flags)
    CARDINALITY=3.392
    ;;

  foodtruck)
    CARDINALITY=2.290
    ;;

  image)
    CARDINALITY=1.236
    ;;

  llog)
    CARDINALITY=1.180
    ;;

  medical)
    CARDINALITY=1.245
    ;;

  ohsumed)
    CARDINALITY=1.663
    ;;

  reuters-k500)
    CARDINALITY=1.462
    ;;

  scene)
    CARDINALITY=1.074
    ;;

  slashdot)
    CARDINALITY=1.181
    ;;

  tmc2007)
    CARDINALITY=2.158
    ;;

  yeast)
    CARDINALITY=4.237
    ;;

  *)
    echo "dataset not supported"
    exit 22

esac

ceiling $CARDINALITY