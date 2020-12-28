import sys

import numpy
import setuptools
from Cython.Build import cythonize

# True, if annotated Cython source files that highlight Python interactions should be created
ANNOTATE = False

# True, if all Cython compiler optimizations should be disabled
DEBUG = False
sources = [
    '**/*.pyx',
    'boomer/common/cpp/data/matrix_dense.cpp',
    'boomer/common/cpp/data/matrix_dok_binary.cpp',
    'boomer/common/cpp/data/vector_dense.cpp',
    'boomer/common/cpp/data/vector_dok_binary.cpp',
    'boomer/common/cpp/data/vector_sparse_array.cpp',
    'boomer/common/cpp/data/vector_mapping_dense.cpp',
    'boomer/common/cpp/indices/index_vector_full.cpp',
    'boomer/common/cpp/indices/index_vector_partial.cpp',
    'boomer/common/cpp/input/feature_matrix_c_contiguous.cpp',
    'boomer/common/cpp/input/feature_matrix_csc.cpp',
    'boomer/common/cpp/input/feature_matrix_csr.cpp',
    'boomer/common/cpp/input/feature_matrix_fortran_contiguous.cpp',
    'boomer/common/cpp/input/feature_vector.cpp',
    'boomer/common/cpp/input/label_matrix_dense.cpp',
    'boomer/common/cpp/input/label_matrix_dok.cpp',
    'boomer/common/cpp/input/nominal_feature_mask_dok.cpp',
    'boomer/common/cpp/model/body_conjunctive.cpp',
    'boomer/common/cpp/model/body_empty.cpp',
    'boomer/common/cpp/model/condition_list.cpp',
    'boomer/common/cpp/sampling/random.cpp',
    'boomer/common/cpp/sampling/feature_sampling_no.cpp',
    'boomer/common/cpp/sampling/feature_sampling_random.cpp',
    'boomer/common/cpp/sampling/instance_sampling_bagging.cpp',
    'boomer/common/cpp/sampling/instance_sampling_no.cpp',
    'boomer/common/cpp/sampling/instance_sampling_random.cpp',
    'boomer/common/cpp/sampling/label_sampling_no.cpp',
    'boomer/common/cpp/sampling/label_sampling_random.cpp',
    'boomer/common/cpp/sampling/weight_vector_dense.cpp',
    'boomer/common/cpp/sampling/weight_vector_equal.cpp',
    'boomer/common/cpp/stopping/stopping_criterion_size.cpp',
    'boomer/common/cpp/stopping/stopping_criterion_time.cpp',
    'boomer/common/cpp/head_refinement/head_refinement_full.cpp',
    'boomer/common/cpp/head_refinement/head_refinement_single.cpp',
    'boomer/common/cpp/head_refinement/prediction.cpp',
    'boomer/common/cpp/head_refinement/prediction_evaluated.cpp',
    'boomer/common/cpp/head_refinement/prediction_full.cpp',
    'boomer/common/cpp/head_refinement/prediction_partial.cpp',
    'boomer/common/cpp/post_processing/post_processor_no.cpp',
    'boomer/common/cpp/pruning/pruning_irep.cpp',
    'boomer/common/cpp/pruning/pruning_no.cpp',
    'boomer/common/cpp/rule_evaluation/score_vector_dense.cpp',
    'boomer/common/cpp/rule_evaluation/score_vector_label_wise_dense.cpp',
    'boomer/common/cpp/rule_refinement/rule_refinement.cpp',
    'boomer/common/cpp/rule_refinement/rule_refinement_exact.cpp',
    'boomer/common/cpp/rule_refinement/rule_refinement_approximate.cpp',
    'boomer/common/cpp/binning/bin_vector.cpp',
    'boomer/common/cpp/binning/feature_binning_equal_frequency.cpp',
    'boomer/common/cpp/binning/feature_binning_equal_width.cpp',
    'boomer/common/cpp/thresholds/coverage_mask.cpp',
    'boomer/common/cpp/thresholds/thresholds_exact.cpp',
    'boomer/common/cpp/thresholds/thresholds_approximate.cpp',
    'boomer/boosting/cpp/data/matrix_dense_numeric.cpp',
    'boomer/boosting/cpp/data/matrix_dense_label_wise.cpp',
    'boomer/boosting/cpp/data/matrix_dense_example_wise.cpp',
    'boomer/boosting/cpp/data/vector_dense_label_wise.cpp',
    'boomer/boosting/cpp/data/vector_dense_example_wise.cpp',
    'boomer/boosting/cpp/losses/loss_label_wise_logistic.cpp',
    'boomer/boosting/cpp/losses/loss_label_wise_squared_error.cpp',
    'boomer/boosting/cpp/losses/loss_example_wise_logistic.cpp',
    'boomer/boosting/cpp/math/blas.cpp',
    'boomer/boosting/cpp/math/lapack.cpp',
    'boomer/boosting/cpp/post_processing/shrinkage_constant.cpp',
    'boomer/boosting/cpp/rule_evaluation/rule_evaluation_example_wise_regularized.cpp',
    'boomer/boosting/cpp/rule_evaluation/rule_evaluation_label_wise_regularized.cpp',
    'boomer/boosting/cpp/statistics/statistics_label_wise_dense.cpp',
    'boomer/boosting/cpp/statistics/statistics_example_wise_dense.cpp',
    'boomer/seco/cpp/head_refinement/head_refinement_partial.cpp',
    'boomer/seco/cpp/head_refinement/lift_function_peak.cpp',
    'boomer/seco/cpp/heuristics/heuristic_f_measure.cpp',
    'boomer/seco/cpp/heuristics/heuristic_hamming_loss.cpp',
    'boomer/seco/cpp/heuristics/heuristic_m_estimate.cpp',
    'boomer/seco/cpp/heuristics/heuristic_precision.cpp',
    'boomer/seco/cpp/heuristics/heuristic_recall.cpp',
    'boomer/seco/cpp/heuristics/heuristic_wra.cpp',
    'boomer/seco/cpp/rule_evaluation/rule_evaluation_label_wise_heuristic.cpp',
    'boomer/seco/cpp/statistics/statistics_label_wise_dense.cpp',
    'boomer/seco/cpp/stopping/stopping_criterion_coverage.cpp'
]

# The compiler/linker argument to enable OpenMP support

COMPILE_FLAG_OPEN_MP = '/openmp' if sys.platform.startswith('win') else '-fopenmp'

extensions = [
    setuptools.Extension(name='*', sources=sources, language='c++', extra_compile_args=[COMPILE_FLAG_OPEN_MP],
                         extra_link_args=[COMPILE_FLAG_OPEN_MP],
                         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
]

compiler_directives = {
    'boundscheck': DEBUG,
    'wraparound': DEBUG,
    'cdivision': not DEBUG,
    'initializedcheck': DEBUG
}

setuptools.setup(name='boomer',
                 version='0.4.0',
                 description='BOOMER - An algorithm for learning gradient boosted multi-label classification rules',
                 url='https://github.com/mrapp-ke/Boomer',
                 author='Michael Rapp',
                 author_email='mrapp@ke.tu-darmstadt.de',
                 license='MIT',
                 packages=['boomer'],
                 install_requires=[
                     "numpy>=1.19.0",
                     "scipy>=1.5.0",
                     "Cython>=0.29.0",
                     'scikit-learn>=0.23.0',
                     'scikit-multilearn>=0.2.0',
                     'liac-arff>=2.5.0',
                     'requests>=2.25.0'
                 ],
                 python_requires='>=3.7',
                 ext_modules=cythonize(extensions, language_level='3', annotate=ANNOTATE,
                                       compiler_directives=compiler_directives),
                 include_dirs=[numpy.get_include()],
                 zip_safe=False)
