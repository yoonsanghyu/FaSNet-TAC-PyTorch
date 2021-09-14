from collections import OrderedDict


# sometimes checkpoint has 'module.' prefixes
def load_model_with_fixes(model, model_info):
    try:
        model.load_state_dict(model_info['model_state_dict'])
    except KeyError:
        state_dict = OrderedDict()
        for k, v in model_info['model_state_dict'].items():
            name = k.replace("module.", "")
            state_dict[name] = v
        model.load_state_dict(state_dict)
