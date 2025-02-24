
class AnyGradVersion(str):
    def __init__(self, version:str):
        self.version = version
    
    def __repr__(self):
        return f"Version(anygrad.{self.version})"
    
__version__ = AnyGradVersion('0.0.1')