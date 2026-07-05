```{image} _static/logo_boomer_light.svg
---
align: center
alt: 'BOOMER: Gradient Boosted Multi-Label Classification Rules'
class: only-light
---
```

```{image} _static/logo_boomer_dark.svg
---
align: center
alt: 'BOOMER: Gradient Boosted Multi-Label Classification Rules'
class: only-dark
---
```

This is a research project evolving around the machine learning algorithm {ref}`BOOMER <user-guide-boomer>` – An algorithm for learning ensembles of gradient boosted multi-output rules that integrates with the popular [scikit-learn](https://scikit-learn.org) machine learning framework. It is aimed at multi-output problems, including [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) and [multi-output regression](https://en.wikipedia.org/wiki/Regression_analysis).

The BOOMER algorithm is build upon a modular framework for implementing rule learning algorithms. This enables to implement different kinds of algorithms more easily. One example is the multi-label {ref}`SeCo algorithm <user-guide-seco>` provided by this project. It is based on traditional rule learning techniques and is particularly well-suited for learning interpretable models. Additional algorithms may follow in the future. The same applies to tools and utilities evolving around these algorithms.

**Software packages provides by this project**

````{grid} 1 1 2 3
```{grid-item-card} BOOMER Algorithm
:link: user-guide-boomer
:link-type: ref
:link-alt: Documentation of the BOOMER algorithm 
:text-align: center

A gradient boosting algorithm for multi-output classification and regression
```

```{grid-item-card} SeCo Algorithm
:link: user-guide-seco
:link-type: ref
:link-alt: Documentation of the SeCo algorithm
:text-align: center

A separate-and-conquer algorithm for multi-label classification
```

```{grid-item-card} mlrl-testbed
:link: user-guide-testbed
:link-type: ref
:link-alt: Documentation of the command line utility mlrl-testbed
:text-align: center

A command line utility for running machine learning experiments
```
````

**Other sources of information**

````{grid} 1 1 2 3
```{grid-item-card} GitHub Repository
:link: https://github.com/mrapp-ke/MLRL-Boomer
:link-alt: The GitHub repository of this project 
:text-align: center
:img-bottom: _static/icon_repository_dark.svg
:class-item: only-dark
```

```{grid-item-card} GitHub Repository
:link: https://github.com/mrapp-ke/MLRL-Boomer
:link-alt: The GitHub repository of this project 
:text-align: center
:img-bottom: _static/icon_repository_light.svg
:class-item: only-light
```

```{grid-item-card} Issue Tracker
:link: https://github.com/mrapp-ke/MLRL-Boomer/issues
:link-alt: The issue tracker of this project
:text-align: center
:img-bottom: _static/icon_issue_tracker_dark.svg
:class-item: only-dark
```

```{grid-item-card} Issue Tracker
:link: https://github.com/mrapp-ke/MLRL-Boomer/issues
:link-alt: The issue tracker of this project
:text-align: center
:img-bottom: _static/icon_issue_tracker_light.svg
:class-item: only-light
```

```{grid-item-card} References
:link: references
:link-type: ref
:link-alt: A list of publications discussing the methodology of our algorithms
:text-align: center
:img-bottom: _static/icon_references_dark.svg
:class-item: only-dark
```

```{grid-item-card} References
:link: references
:link-type: ref
:link-alt: A list of publications discussing the methodology of our algorithms
:text-align: center
:img-bottom: _static/icon_references_light.svg
:class-item: only-light
```
````

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
user_guide/foundations/index
user_guide/boosting/index
user_guide/seco/index
user_guide/testbed/index
user_guide/references
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
developer_guide/continuous_integration
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
misc/CHANGELOG
misc/CONTRIBUTORS
misc/CODE_OF_CONDUCT
misc/LICENSE
Source Code <https://github.com/mrapp-ke/MLRL-Boomer>
Issue Tracker <https://github.com/mrapp-ke/MLRL-Boomer/issues>
```
