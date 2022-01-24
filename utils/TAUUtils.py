import yaml

def load_yaml(file_path):
    return yaml.load(open(file_path, 'r'), Loader=yaml.SafeLoader)