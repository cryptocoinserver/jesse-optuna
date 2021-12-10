import copy

import optuna
import joblib
import numpy as np

class JoblibStudy:
    def __init__(self, **study_parameters):
        self.study_parameters = study_parameters
        self.study: optuna.study.Study = optuna.create_study(**study_parameters)

    def _optimize_study(self, func, n_trials, **optimize_parameters):
        study_parameters = copy.copy(self.study_parameters)
        study_parameters["study_name"] = self.study.study_name
        study_parameters["load_if_exists"] = True
        study = optuna.create_study(**study_parameters)
        study.optimize(func, n_trials=n_trials, **optimize_parameters, catch=(Exception,))


    @staticmethod
    def _split_trials(n_trials, n_jobs):
        n_per_job, remaining = divmod(n_trials, n_jobs)
        for _ in range(n_jobs):
            yield n_per_job + (1 if remaining > 0 else 0)
            remaining -= 1

    def optimize(self, func, n_trials=1, n_jobs=-1, **optimize_parameters):
        if n_jobs == -1:
            n_jobs = joblib.cpu_count()

        if n_jobs == 1:
            self.study.optimize(n_trials=n_trials, **optimize_parameters)
        else:
            parallel = joblib.Parallel(n_jobs, verbose=10, max_nbytes=None)
            parallel(
                joblib.delayed(self._optimize_study)(func, n_trials=n_trials_i, **optimize_parameters)
                for n_trials_i in self._split_trials(n_trials, n_jobs)
            )

    def set_user_attr(self, key: str, value):
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        self.study.set_user_attr(key, value)

    def __getattr__(self, name):
        if not name.startswith("_") and hasattr(self.study, name):
            return getattr(self.study, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")