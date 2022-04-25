# Jesse optuna

Only works with the new GUI version of jesse.
Use optuna directly to work with the results and create [visualisations](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html) of the results. Jupyter Notebook might be usefull here.

You need the study name and storage, which you set in the config.yml, to load it for that:

```
    study_name = f"{cfg['strategy_name']}-{cfg['exchange']}-{cfg['symbol']}-{cfg['timeframe']}"
    storage = f"postgresql://{cfg['postgres_username']}:{cfg['postgres_password']}@{cfg['postgres_host']}:{cfg['postgres_port']}/{cfg['postgres_db_name']}"
```

The config.yml should be self-explainatory.

# Installation

```sh
# install from git
pip install git+https://github.com/cryptocoinserver/jesse-optuna.git

# cd in your Jesse project directory

# create the config file
jesse-optuna create-config

# create the database for optuna 
jesse-optuna create-db optuna_db

# edit the created yml file in your project directory 

# run
jesse-optuna run

```


## Disclaimer
This software is for educational purposes only. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS. Do not risk money which you are afraid to lose. There might be bugs in the code - this software DOES NOT come with ANY warranty.
