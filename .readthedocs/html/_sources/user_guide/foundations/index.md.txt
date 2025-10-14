(user-guide-foundations)=

# Foundations

```{seealso}
Most of the content in this chapter has been taken from the Ph.D. thesis {ref}`‘Multi-label Rule Learning’, Michael Rapp (2022) <references-rapp2022phdthesis>`.
```

Among the most common machine learning approaches, one may distinguish between [statistical](https://en.wikipedia.org/wiki/Statistical_learning_theory) and [symbolic](https://en.wikipedia.org/wiki/Symbolic_artificial_intelligence) methods. The former use statistical optimization techniques to determine the parameters of a predictive function. Examples include [artificial neural networks](<https://en.wikipedia.org/wiki/Neural_network_(machine_learning)>), [support vector machines](https://en.wikipedia.org/wiki/Support_vector_machine), or [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression). Symbolic learning methods rely on symbolic descriptions to represent learned concepts and capture knowledge about a problem domain. For example, in [decision tree learning](https://en.wikipedia.org/wiki/Decision_tree_learning) or [rule learning](https://en.wikipedia.org/wiki/Rule_induction), which this project focuses on primarily, models are typically represented in terms of logical *if*-*then*-clauses that test for the properties of given examples to determine a prediction. In the following, we discuss the general methodology of these machine learning methods, as well as the conceptual framework used by our implementations:

```{toctree}
---
maxdepth: 1
---
rule_learning
problem_definition
framework
```
