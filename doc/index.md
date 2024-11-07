```{image} _static/logo_light.svg
---
align: center
alt: 'BOOMER: Gradient Boosted Multi-Label Classification Rules'
class: only-light
---
```

```{image} _static/logo_dark.svg
---
align: center
alt: 'BOOMER: Gradient Boosted Multi-Label Classification Rules'
class: only-dark
---
```

BOOMER is an algorithm for learning ensembles of gradient boosted multi-output rules that integrates with the popular [scikit-learn](https://scikit-learn.org) machine learning framework. It allows to train a machine learning model on labeled training data, which can afterward be used to make predictions for unseen data. In contrast to prominent boosting algorithms like [XGBoost](https://xgboost.readthedocs.io/en/latest/) or [LightGBM](https://lightgbm.readthedocs.io/en/latest/), the algorithm is aimed at multi-output problems. On the one hand, this includes [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) problems, where individual data examples do not only correspond to a single class, but may be associated with several labels at the same time. Real-world applications of this problem domain include the assignment of keywords to text documents, the annotation of multimedia data, such as images, videos or audio recordings, as well as applications in the field of biology, chemistry and more. On the other hand, multi-output [regression](https://en.wikipedia.org/wiki/Regression_analysis) problems require to predict for more than a single numerical output variable.

To provide a versatile tool for different use cases, great emphasis is put on the *efficiency* of the implementation. Moreover, to ensure its *flexibility*, it is designed in a modular fashion and can therefore easily be adjusted to different requirements. This modular approach enables implementing different kind of rule learning algorithms. For example, this project does also provide a Separate-and-Conquer (SeCo) algorithm based on traditional rule learning techniques that are particularly well-suited for learning interpretable models.

This document is intended for end users of our algorithms and developers who are interested in their implementation. In addition, the following links might be of interest:

- For a detailed description of the methodology used by the algorithms, please refer to the {ref}`list of publications <references>`.
- The source code maintained by this project can be found in the [GitHub repository](https://github.com/mrapp-ke/MLRL-Boomer).
- Issues with the software, feature requests, or questions to the developers should be posted via the project's [issue tracker](https://github.com/mrapp-ke/MLRL-Boomer/issues).

```{toctree}
---
caption: Quickstart
hidden: true
maxdepth: 2
---
quickstart/installation
quickstart/usage
quickstart/testbed
```

```{toctree}
---
caption: User Guide
hidden: true
maxdepth: 2
---
user_guide/boosting/index
user_guide/seco/index
user_guide/testbed/index
```

```{toctree}
---
caption: Developer Guide
hidden: true
maxdepth: 2
---
developer_guide/project_structure
developer_guide/compilation
developer_guide/documentation
developer_guide/coding_standards
developer_guide/api/python/index
developer_guide/api/cpp/index
```

```{toctree}
---
caption: Further Information
hidden: true
maxdepth: 2
---
misc/references
misc/CHANGELOG
misc/CONTRIBUTORS
misc/CODE_OF_CONDUCT
misc/LICENSE
```
