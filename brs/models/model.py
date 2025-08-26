from abc import ABC, abstractmethod
from typing import Any
import pathlib

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: Any, y:Any=None) -> "BaseModel":
        pass

    @abstractmethod
    def predict(self, X:Any) -> Any:
        pass

    @abstractmethod
    def save(self, path: pathlib.Path):
        pass

    @abstractmethod
    def load(self, path: pathlib.Path) -> "BaseModel":
        pass
