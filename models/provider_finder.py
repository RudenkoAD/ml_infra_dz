from typing import Optional, Type

from models.providers.base_provider import Provider
from models.providers.huggingface import HuggingFaceProvider
from models.providers.openrouter import OpenRouterProvider

providers: dict[str, Type[Provider]] = {
    "openrouter": OpenRouterProvider,
    "huggingface": HuggingFaceProvider
}
    

def get_provider(provider: str, api_key: Optional[str] = None, model_name: str = "deepseek/deepseek-r1-distill-qwen-7b") -> Provider:
    """
    Factory function to get the appropriate provider instance based on the provider name.
    
    Args:
        provider (str): The name of the provider (e.g., "openrouter", "huggingface").
        api_key (str, optional): API key for the provider. Defaults to None.
        model_name (str, optional): Model name to use with the provider. Defaults to "deepseek/deepseek-r1-distill-qwen-7b".
    
    Returns:
        An instance of the specified provider class.
    
    Raises:
        ValueError: If the provider is not recognized.
    """
    if provider not in providers:
        raise ValueError(f"Provider '{provider}' is not recognized.")
    
    return providers[provider](api_key=api_key, model_name=model_name)
