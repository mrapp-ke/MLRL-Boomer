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

BOOMER is an algorithm for learning ensembles of gradient boosted multi-label classification rules that integrates with the popular [scikit-learn](https://scikit-learn.org) machine learning framework. It allows to train a machine learning model on labeled training data, which can afterwards be used to make predictions for unseen data. In contrast to prominent boosting algorithms like [XGBoost](https://xgboost.readthedocs.io/en/latest/) or [LightGBM](https://lightgbm.readthedocs.io/en/latest/), the algorithm is aimed at [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) problems, where individual data examples do not only correspond to a single class, but may be associated with several labels at the same time. Real-world applications of multi-label classification include the assignment of keywords to text documents, the annotation of multimedia data, such as images, videos or audio recordings, as well as applications in the field of biology, chemistry and more.

To provide a versatile tool for different use cases, great emphasis is put on the *efficiency* of the implementation. Moreover, to ensure its *flexibility*, it is designed in a modular fashion and can therefore easily be adjusted to different requirements. This modular approach enables implementing different kind of rule learning algorithms. For example, this project does also provide a Separate-and-Conquer (SeCo) algorithm based on traditional rule learning techniques that are particularly well-suited for learning interpretable models.

This document is intended for users and developers who are interested in the algorithm's implementation. For a detailed description of the used methodology, please refer to the publications that are listed under {ref}`references`.

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
