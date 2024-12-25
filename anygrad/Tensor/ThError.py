
def broadcast_error(allow:bool, msg:str):
    if allow == False:
        raise ValueError("The Shape is not broadcast for" + msg)
    
def sum_error(allow:bool, msg:str):
    if not allow:
        raise ValueError("The Axis is incompatable for" + msg)
