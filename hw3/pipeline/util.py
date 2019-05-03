def wrap_list(maybe_list):
    if isinstance(maybe_list, list):
        return maybe_list
    else:
        return [maybe_list]
