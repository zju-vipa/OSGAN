from ruamel.yaml import YAML

def load_yaml(filepath):
    yaml=YAML()  
    with open(filepath, 'r') as f:
        return yaml.load(f)


def flatten_dict(dic):
    flattned = dict()

    def _flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                if prefix is None:
                    _flatten(k, v)
                else:
                    _flatten(prefix+'/%s'%k, v)
            else:
                if prefix is None:
                    flattned[k] = v
                else:
                    flattned[ prefix+'/%s'%k ] = v
        
    _flatten(None, dic)
    return flattned
