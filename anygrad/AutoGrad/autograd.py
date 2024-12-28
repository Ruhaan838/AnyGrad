
class GradMode:
    _enable = True
    @classmethod
    def is_enabled(cls):
        return cls._enable
    
    @classmethod
    def set_enable(cls, mode:bool):
        cls._enable = mode
