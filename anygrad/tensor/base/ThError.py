def broadcast_error(allow: bool, msg: str):
    if not allow:
        raise ValueError("The Shape is not broadcast for" + msg)


def sum_error(allow: bool, msg: str):
    if not allow:
        raise ValueError("The Axis is incompatable for" + msg)


def dim_error(allow: bool, msg: str):
    if not allow:
        raise ValueError("The Ndim is incopatable for" + msg)

def view_error(allow: bool, msg: str):
    if not allow:
        raise ValueError("The size is incopetable for" + msg)