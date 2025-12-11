"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow reading datasets from SVM (light) files.
"""
from functools import cached_property
from pathlib import Path
from typing import List, Optional, override

import numpy as np

from scipy.sparse import sparray
from sklearn.datasets import load_svmlight_file

from mlrl.common.cython.prediction import csr_array

from mlrl.testbed_sklearn.experiments.dataset import Attribute, AttributeType, TabularDataset

from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.input.data import DatasetInputData
from mlrl.testbed.experiments.input.sources.source import DatasetFileSource
from mlrl.testbed.experiments.problem_domain import ProblemDomain, RegressionProblem
from mlrl.testbed.experiments.state import ExperimentState


class SvmFileSource(DatasetFileSource):
    """
    Allows to read a dataset from an SVM (light) file.
    """

    SUFFIX_SVM = 'svm'

    class SvmFile:
        """
        Provides access to the content of an SVM file.
        """

        def __init__(self, feature_matrix: sparray, output_matrix: sparray):
            """
            :param feature_matrix:  The feature matrix that is stored in the file
            :param output_matrix:   The ground truth matrix that is stored in the file
            """
            self.feature_matrix = feature_matrix
            self.output_matrix = output_matrix

        @staticmethod
        def from_file(file_path: Path, problem_domain: ProblemDomain) -> 'SvmFileSource.SvmFile':
            """
            Loads the content of an SVM file.

            :param file_path:       The path to the SVM file
            :param problem_domain:  The problem domain, the SVM file is concerned with
            :return:                A `SvmFileSource.SvmFile` that has been loaded
            """
            # pylint: disable=unbalanced-tuple-unpacking,useless-suppression
            feature_matrix, output_matrix_rows = load_svmlight_file(file_path,
                                                                    dtype=problem_domain.feature_dtype,
                                                                    multilabel=True)
            indptr: List[int] = []
            indices: List[int] = []

            if isinstance(problem_domain, RegressionProblem):
                values: List[float] = []

                for row in output_matrix_rows:
                    indptr.append(len(indices))
                    indices.extend(index for index in range(len(row)))
                    values.extend(map(float, row))

                indptr.append(len(indices))
                data = np.asarray(values).astype(problem_domain.output_dtype)
            else:
                for row in output_matrix_rows:
                    indptr.append(len(indices))
                    indices.extend(map(int, row))

                indptr.append(len(indices))
                data = np.ones(shape=len(indices), dtype=problem_domain.output_dtype)

            output_matrix = csr_array((data, np.asarray(indices), np.asarray(indptr)))
            return SvmFileSource.SvmFile(feature_matrix=feature_matrix, output_matrix=output_matrix)

        @cached_property
        def features(self) -> List[Attribute]:
            """
            A list that stores all features contained in the SVM file.
            """
            num_features = self.feature_matrix.shape[1]
            return [
                Attribute(name=f'Feature {feature_index + 1}', attribute_type=AttributeType.NUMERICAL)
                for feature_index in range(num_features)
            ]

        @cached_property
        def outputs(self) -> List[Attribute]:
            """
            A list that stores all outputs contained in the SVM file.
            """
            num_features = self.feature_matrix.shape[1]
            num_outputs = self.output_matrix.shape[1]
            return [
                Attribute(name=f'Output {num_features + output_index + 1}', attribute_type=AttributeType.NOMINAL)
                for output_index in range(num_outputs)
            ]

    class SvmDataset:
        """
        Provides access to the content of an SVM file.
        """

        def __init__(self, svm_file: 'SvmFileSource.SvmFile'):
            """
            :param svm_file: The content of the SVM file
            """
            self.svm_file = svm_file

        @staticmethod
        def from_file(svm_file: 'SvmFileSource.SvmFile') -> 'SvmFileSource.SvmDataset':
            """
            Creates and returns an SVM dataset from given SVM file.

            :param svm_file:    The content of the SVM file
            :return:            The SVM dataset that has been created
            """
            return SvmFileSource.SvmDataset(svm_file)

        @property
        def features(self) -> List[Attribute]:
            """
            A list that stores all features contained in the dataset.
            """
            return self.svm_file.features

        @property
        def outputs(self) -> List[Attribute]:
            """
            A list that stores all outputs contained in the dataset.
            """
            return self.svm_file.outputs

        @property
        def feature_matrix(self) -> sparray:
            """
            The feature matrix contained in the dataset.
            """
            return self.svm_file.feature_matrix

        @property
        def output_matrix(self) -> sparray:
            """
            The output matrix contained in the dataset.
            """
            return self.svm_file.output_matrix

    def __init__(self, directory: Path):
        """
        :param directory: The path to the directory of the file
        """
        super().__init__(directory=directory, suffix=self.SUFFIX_SVM)

    @override
    def _read_dataset_from_file(self, state: ExperimentState, file_path: Path,
                                _: DatasetInputData) -> Optional[Dataset]:
        problem_domain = state.problem_domain
        svm_file = SvmFileSource.SvmFile.from_file(file_path, problem_domain=problem_domain)
        svm_dataset = SvmFileSource.SvmDataset.from_file(svm_file)
        return TabularDataset(x=svm_dataset.feature_matrix,
                              y=svm_dataset.output_matrix,
                              features=svm_dataset.features,
                              outputs=svm_dataset.outputs)
