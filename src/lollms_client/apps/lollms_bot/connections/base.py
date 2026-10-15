from abc import ABC, abstractmethod
from typing import Dict, Any

class Connection(ABC):
    def __init__(self, config: Dict[str, Any], llm_config: Dict[str, Any]):
        self.config = config
        self.llm_config = llm_config

    @abstractmethod
    async def run(self):
        """Start the connection client."""
        pass