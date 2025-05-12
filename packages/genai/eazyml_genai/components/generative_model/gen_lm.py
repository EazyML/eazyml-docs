from abc import ABC, abstractmethod

class LanguageModel(ABC):

    def __init__(self, **kwargs):
        self.__provider = kwargs.get('provider', None)
        self.__provider = kwargs.get('model_name', None)

    @property
    def provider(self):
        return self.__provider
    
    @property
    def model_name(self):
        return self.__model_name
