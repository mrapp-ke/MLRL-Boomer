.. _pre_processing:

Data Pre-Processing
===================

Depending on the dataset at hand, it might be desirable to apply pre-processing techniques to the data before training a machine learning model. The pre-processing techniques that are supported are discussed in the following. When using such a technique, it will be applied to the training and test sets (see :ref:`evaluation`), before training a model and querying it for predictions, respectively.

One-Hot-Encoding
----------------

.. warning::
    When using the algorithms provided by this project, the use of one-hot-encoding is typically not adviced, as they can deal with nominal and binary features in a more efficient way. However, as argued below, it might still be useful for a fair comparison with machine learning approaches that cannot deal with such features.

Not all machine learning methods can deal with nominal or binary features out-of-the-box. In such cases, it is necessary to pre-process the available data in order to convert these features into numerical ones. The most commonly used technique for this purpose is referred to as `one-hot-encoding <https://en.wikipedia.org/wiki/One-hot>`__. It will replace each feature that comes with a predefined set of discrete values, with several numerical features corresponding to each of the potential values. The values for these newly added features will be set to ``1``, if an original data point was associated with the corresponding nominal value, or ``0`` otherwise. Because the resulting dataset will typically entail more features than the original one, the use of one-hot-encoding often increases the computational costs and time needed for training a machine learning model.

Even though nominal and binary features are natively supported in an efficient way by all algorithms provided by this project, it might still be useful to use one-hot-encoding if one seeks for a fair comparison with machine learning approaches that cannot deal with such features. In such cases, you can provide the argument ``--one-hot-encoding true`` to the command line API:

.. code-block:: text

   boomer --data-dir /path/to/datsets/ --dataset dataset-name --one-hot-encoding true

Under the hood, the program will make use of scikit-learn's `OneHotEncoder <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html>`__ for pre-processing the data.
