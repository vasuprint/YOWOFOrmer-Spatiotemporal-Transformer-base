import yaml
import argparse

def build_config(config_file=None):
    # Parse command line arguments if no config file is provided
    if config_file is None:
        parser = argparse.ArgumentParser(description='YOWOFormer Training')
        parser.add_argument('-cf', '--config-file', type=str, required=True,
                            help='Path to configuration file')
        args = parser.parse_args()
        config_file = args.config_file

    with open(config_file, "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    if config.get('active_checker', False):
        pass

    return config