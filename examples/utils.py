import torch
import os

def load_orig_task_weights(model,weights_path=None):
    weights = torch.load(weights_path)
    res_weights = {}
    print("Initializing new task weights with old task. Loading pretrained weights...")
    for name in model.state_dict().keys():
        if '_new_task' in name:
            foo = name.replace('_new_task','')
            res_weights[name] = weights[foo]
            continue
        res_weights[name] = weights[name]
    model.load_state_dict(res_weights)
    return model

def xavier_init(model):
    old_weights = model.state_dict()
    for name, weights in model.named_parameters():
        if 'weight' in name and 'bn' not in name:
            old_weights[name] = torch.nn.init.xavier_uniform(old_weights[name])
    model.load_state_dict(old_weights)
    return model

def get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
        )
    )
    return torch_home
