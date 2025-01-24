"""this file implements mock evaluation pipeline """

import os
import sys
import pandas as pd
from typing import Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(PROJECT_DIR)

from evaluation_pipeline import EvaluationPipeline


class MockEvaluationPipeline(EvaluationPipeline):
    """a mock for evaluation pipeline"""

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        models: list,
        kfold=False,
        n_splits: Optional[int] = 4,
    ) -> None:
        super().__init__(data, target_column, models, kfold, n_splits)

    def create_report(self):
        """mock for creating report"""
        pass
