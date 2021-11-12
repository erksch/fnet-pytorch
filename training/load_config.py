import os
from configparser import ConfigParser

from tabulate import tabulate

def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise Exception('configuration file {} does not exist'.format(config_path))

    configparser = ConfigParser()
    configparser.read(config_path)
    
    config = {}

    # == general ==
    config['experiment_name'] = configparser.get('general', 'experiment_name', fallback='unnamed')
    config['gpu_id'] = configparser.getint('general', 'gpu_id', fallback=-1)

    # == model ==
    config['fnet_config'] = configparser.get('model', 'fnet_config')
    config['fnet_checkpoint'] = configparser.get('model', 'fnet_checkpoint')

    # == training ==
    config['learning_rate'] = configparser.getfloat('training', 'learning_rate')
    config['train_batch_size'] = configparser.getint('training', 'train_batch_size')
    config['eval_batch_size'] = configparser.getint('training', 'eval_batch_size')
    config['eval_frequency'] = configparser.getint('training', 'eval_frequency')
    config['eval_steps'] = configparser.getint('training', 'eval_steps')

    # == tokenizer ==
    config['tokenizer'] = {}
    config['tokenizer']['type'] = configparser.get('tokenizer', 'type')
    config['tokenizer']['vocab'] = configparser.get('tokenizer', 'vocab')
    config['tokenizer']['hf_name'] = configparser.get('tokenizer', 'hf_name')

    return config


def print_config(config):
    print(tabulate(config.items(), tablefmt='grid'))
