Welcome to BOOMER's documentation!
==================================

BOOMER is an algorithm for learning gradient boosted multi-label classification rules. It allows to train a machine learning model on labeled training data, which can afterwards be used to make predictions for unseen data. In contrast to prominent boosting algorithms like `XGBoost <https://xgboost.readthedocs.io/en/latest/>`_ or `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`_, the algorithm is aimed at multi label classification problems, where individual data examples are not only associated with a single class, but may correspond to several labels at the same time.

This document is intented for users and developers that are interested in the algorithm's implementation. For a detailed description of the used methodology, please refer to the following `paper <https://link.springer.com/chapter/10.1007/978-3-030-67664-3_8>`_. A preprint version is publicly available on `arxiv.org <https://arxiv.org/pdf/2006.13346.pdf>`__.

*Rapp M., Loza Mencía E., Fürnkranz J., Nguyen VL., Hüllermeier E. (2020) Learning Gradient Boosted Multi-label Classification Rules. In: Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2020. Lecture Notes in Computer Science, pp. 124-140, vol 12459. Springer, Cham*

Gradient-based label binning (GBLB), which is an extension to the original algorithm, was proposed in the following paper. A preprint version is available on `arxiv.org <https://arxiv.org/pdf/2106.11690.pdf>`__.

*Rapp M., Loza Mencía E., Fürnkranz J., Hüllermeier E. (2021) Gradient-based Label Binning in Multi-label Classification. In: Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2021. Lecture Notes in Computer Science, vol 12977. Springer, Cham*

If you use the algorithm in a scientific publication, we would appreciate citations to the mentioned papers.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart/index

   api/index
