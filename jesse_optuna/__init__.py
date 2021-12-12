import logging
import os
import pathlib
import pickle
import shutil
import traceback

import click
import jesse.helpers as jh
import numpy as np
import optuna
import pkg_resources
import yaml
from jesse.research import backtest, get_candles
from .JoblilbStudy import JoblibStudy


logger = logging.getLogger()
logger.addHandler(logging.FileHandler("optuna.log", mode="w"))

optuna.logging.enable_propagation()

# create a Click group
@click.group()
@click.version_option(pkg_resources.get_distribution("jesse-optuna").version)
def cli() -> None:
    pass


@cli.command()
def create_config() -> None:
    validate_cwd()
    target_dirname = pathlib.Path().resolve()
    package_dir = pathlib.Path(__file__).resolve().parent
    shutil.copy2(f'{package_dir}/optuna_config.yml', f'{target_dirname}/optuna_config.yml')

@cli.command()
@click.argument('db_name', required=True, type=str)
def create_db(db_name: str) -> None:
    validate_cwd()
    cfg = get_config()
    import psycopg2

    # establishing the connection
    conn = psycopg2.connect(
        database="postgres", user=cfg['postgres_username'], password=cfg['postgres_password'], host=cfg['postgres_host'], port=cfg['postgres_port']
    )
    conn.autocommit = True

    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    # Creating a database
    cursor.execute('CREATE DATABASE ' + str(db_name))
    print(f"Database {db_name} created successfully........")

    # Closing the connection
    conn.close()


@cli.command()
def run() -> None:
    validate_cwd()

    cfg = get_config()
    study_name = f"{cfg['strategy_name']}-{cfg['exchange']}-{cfg['symbol']}-{cfg['timeframe']}"
    storage = f"postgresql://{cfg['postgres_username']}:{cfg['postgres_password']}@{cfg['postgres_host']}:{cfg['postgres_port']}/{cfg['postgres_db_name']}"

    sampler = optuna.samplers.NSGAIISampler(population_size=cfg['population_size'], mutation_prob=cfg['mutation_prob'],
                                            crossover_prob=cfg['crossover_prob'], swapping_prob=cfg['swapping_prob'])

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    try:
        study = JoblibStudy(study_name=study_name, direction="maximize", sampler=sampler,
                                    storage=storage, load_if_exists=False)
    except optuna.exceptions.DuplicatedStudyError:
        if click.confirm('Previous study detected. Do you want to resume?', default=True):
            study = JoblibStudy(study_name=study_name, direction="maximize", sampler=sampler,
                                        storage=storage, load_if_exists=True)
        elif click.confirm('Delete previous study and start new?', default=False):
            optuna.delete_study(study_name=study_name, storage=storage)
            study = JoblibStudy(study_name=study_name, direction="maximize", sampler=sampler,
                                        storage=storage, load_if_exists=False)
        else:
            print("Exiting.")
            exit(1)

    study.set_user_attr("strategy_name", cfg['strategy_name'])
    study.set_user_attr("exchange", cfg['exchange'])
    study.set_user_attr("symbol", cfg['symbol'])
    study.set_user_attr("timeframe", cfg['timeframe'])


    study.optimize(objective, n_jobs=cfg['n_jobs'], n_trials=cfg['n_trials'])


def get_config():
    cfg_file = pathlib.Path('optuna_config.yml')

    if not cfg_file.is_file():
        print("optuna_config.yml not found. Run create-config command.")
        exit()
    else:
        with open("optuna_config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)

    return cfg


def objective(trial):
    cfg = get_config()

    StrategyClass = jh.get_strategy_class(cfg['strategy_name'])
    hp_dict = StrategyClass().hyperparameters()

    for st_hp in hp_dict:
        if st_hp['type'] is int:
            if 'step' not in st_hp:
                st_hp['step'] = 1
            trial.suggest_int(st_hp['name'], st_hp['min'], st_hp['max'], step=st_hp['step'])
        elif st_hp['type'] is float:
            if 'step' not in st_hp:
                st_hp['step'] = 0.1
            trial.suggest_float(st_hp['name'], st_hp['min'], st_hp['max'], step=st_hp['step'])
        elif st_hp['type'] is bool:
            trial.suggest_categorical(st_hp['name'], [True, False])
        else:
            raise TypeError('Only int, bool and float types are implemented for strategy parameters.')

    try:
        training_data_metrics = backtest_function(cfg['timespan-train']['start_date'],
                                                  cfg['timespan-train']['finish_date'],
                                                  trial.params, cfg)
    except Exception as err:
        logger.error("".join(traceback.TracebackException.from_exception(err).format()))
        raise err


    if training_data_metrics is None:
        return np.nan


    if training_data_metrics['total'] <= 5:
        return np.nan

    total_effect_rate = np.log10(training_data_metrics['total']) / np.log10(cfg['optimal-total'])
    total_effect_rate = min(total_effect_rate, 1)
    ratio_config = cfg['fitness-ratio']
    if ratio_config == 'sharpe':
        ratio = training_data_metrics['sharpe_ratio']
        ratio_normalized = jh.normalize(ratio, -.5, 5)
    elif ratio_config == 'calmar':
        ratio = training_data_metrics['calmar_ratio']
        ratio_normalized = jh.normalize(ratio, -.5, 30)
    elif ratio_config == 'sortino':
        ratio = training_data_metrics['sortino_ratio']
        ratio_normalized = jh.normalize(ratio, -.5, 15)
    elif ratio_config == 'omega':
        ratio = training_data_metrics['omega_ratio']
        ratio_normalized = jh.normalize(ratio, -.5, 5)
    elif ratio_config == 'serenity':
        ratio = training_data_metrics['serenity_index']
        ratio_normalized = jh.normalize(ratio, -.5, 15)
    elif ratio_config == 'smart sharpe':
        ratio = training_data_metrics['smart_sharpe']
        ratio_normalized = jh.normalize(ratio, -.5, 5)
    elif ratio_config == 'smart sortino':
        ratio = training_data_metrics['smart_sortino']
        ratio_normalized = jh.normalize(ratio, -.5, 15)
    else:
        raise ValueError(
            f'The entered ratio configuration `{ratio_config}` for the optimization is unknown. Choose between sharpe, calmar, sortino, serenity, smart shapre, smart sortino and omega.')
    if ratio < 0:
        return np.nan

    score = total_effect_rate * ratio_normalized

    try:
        testing_data_metrics = backtest_function(cfg['timespan-testing']['start_date'], cfg['timespan-testing']['finish_date'], trial.params, cfg)
    except Exception as err:
        logger.error("".join(traceback.TracebackException.from_exception(err).format()))
        raise err

    if testing_data_metrics is None:
        return np.nan

    for key, value in testing_data_metrics.items():
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        trial.set_user_attr(f"testing-{key}", value)

    for key, value in training_data_metrics.items():
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        trial.set_user_attr(f"training-{key}", value)
    return score

def validate_cwd() -> None:
    """
    make sure we're in a Jesse project
    """
    ls = os.listdir('.')
    is_jesse_project = 'strategies' in ls and 'storage' in ls

    if not is_jesse_project:
        print('Current directory is not a Jesse project. You must run commands from the root of a Jesse project.')
        exit()


def get_candles_with_cache(exchange: str, symbol: str, start_date: str, finish_date: str) -> np.ndarray:
    path = pathlib.Path('storage/jesse-optuna')
    path.mkdir(parents=True, exist_ok=True)

    cache_file_name = f"{exchange}-{symbol}-1m-{start_date}-{finish_date}.pickle"
    cache_file = pathlib.Path(f'storage/jesse-optuna/{cache_file_name}')

    if cache_file.is_file():
        with open(f'storage/jesse-optuna/{cache_file_name}', 'rb') as handle:
            candles = pickle.load(handle)
    else:
        candles = get_candles(exchange, symbol, '1m', start_date, finish_date)
        with open(f'storage/jesse-optuna/{cache_file_name}', 'wb') as handle:
            pickle.dump(candles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return candles


def backtest_function(start_date, finish_date, hp, cfg):

    candles = {}
    extra_routes = []
    if len(extra_routes) != 0:
        for extra_route in cfg['extra_routes'].items():
            extra_route = extra_route[1]
            candles[jh.key(extra_route['exchange'], extra_route['symbol'])] = {
                'exchange': extra_route['exchange'],
                'symbol': extra_route['symbol'],
                'candles': get_candles_with_cache(
                    extra_route['exchange'],
                    extra_route['symbol'],
                    start_date,
                    finish_date,
                ),
            }
            extra_routes.append({'exchange': extra_route['exchange'], 'symbol': extra_route['symbol'],
                                 'timeframe': extra_route['timeframe']})
    candles[jh.key(cfg['exchange'], cfg['symbol'])] = {
        'exchange': cfg['exchange'],
        'symbol': cfg['symbol'],
        'candles': get_candles_with_cache(
            cfg['exchange'],
            cfg['symbol'],
            start_date,
            finish_date,
        ),
    }

    route = [{'exchange': cfg['exchange'], 'strategy': cfg['strategy_name'], 'symbol': cfg['symbol'],
              'timeframe': cfg['timeframe']}]

    config = {
        'starting_balance': cfg['starting_balance'],
        'fee': cfg['fee'],
        'futures_leverage': cfg['futures_leverage'],
        'futures_leverage_mode': cfg['futures_leverage_mode'],
        'exchange': cfg['exchange'],
        'settlement_currency': cfg['settlement_currency'],
        'warm_up_candles': cfg['warm_up_candles']
    }


    backtest_data = backtest(config, route, extra_routes, candles, True, hp)

    if backtest_data['total'] == 0:
        backtest_data = {'total': 0, 'total_winning_trades': None, 'total_losing_trades': None,
                         'starting_balance': None, 'finishing_balance': None, 'win_rate': None,
                         'ratio_avg_win_loss': None, 'longs_count': None, 'longs_percentage': None,
                         'shorts_percentage': None, 'shorts_count': None, 'fee': None, 'net_profit': None,
                         'net_profit_percentage': None, 'average_win': None, 'average_loss': None, 'expectancy': None,
                         'expectancy_percentage': None, 'expected_net_profit_every_100_trades': None,
                         'average_holding_period': None, 'average_winning_holding_period': None,
                         'average_losing_holding_period': None, 'gross_profit': None, 'gross_loss': None,
                         'max_drawdown': None, 'annual_return': None, 'sharpe_ratio': None, 'calmar_ratio': None,
                         'sortino_ratio': None, 'omega_ratio': None, 'serenity_index': None, 'smart_sharpe': None,
                         'smart_sortino': None, 'total_open_trades': None, 'open_pl': None, 'winning_streak': None,
                         'losing_streak': None, 'largest_losing_trade': None, 'largest_winning_trade': None,
                         'current_streak': None}

    return backtest_data
