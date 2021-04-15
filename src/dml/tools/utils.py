from pathlib import Path

import configparser

def get_project_root():
    cwd = Path.cwd()
    root_idx = cwd.parts.index('dml')
    root_directory = Path(*cwd.parts[:root_idx + 1])
    return root_directory


def get_path_to_file(filename):
    root = get_project_root()
    file_path = list(root.rglob(filename))

    if len(file_path) == 0:
        return 0
    else:
        return file_path[0]


def get_settings(config_file):
    filename = get_path_to_file(config_file)

    config = configparser.ConfigParser()
    config.read(filename)

    model = config.getint('DGP', 'model')
    correlation = config.getfloat('DGP', 'base_correlation')
    K = config.getint('DGP', 'number_of_covariates')
    n = config.getint('DGP', 'sample_size')
    linear = config.getboolean('DGP', 'linear')
    treatment_coef = config.getfloat('DGP', 'treatment_coef')

    methods_str = config.get('DML', 'methods')
    DML2 = config.getboolean('DML', 'DML2')
    splits = config.getint('DML', 'dml_splits')
    sample_splits = config.getint('DML', 'sample_splits')

    seed_str = config.get('others', 'seed')
    seed = None if seed_str == 'None' else int(seed_str)

    methods = ['OLS'] + [method for method in methods_str.strip('[, ]').split(', ')]

    settings = {
        'model': model,
        'treatment_coef': treatment_coef,
        'corr': correlation,
        'k': K,
        'n': n,
        'linear': linear,
        'methods': methods,
        'DML2': DML2,
        'n_splits': splits,
        'n_folds': sample_splits,
        'seed': seed
    }

    return settings
