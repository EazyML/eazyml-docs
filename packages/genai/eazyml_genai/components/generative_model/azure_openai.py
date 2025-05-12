"""
Azure-hosted OpenAI language model integration for the EazyML GenAI framework.

This class initializes an Azure OpenAI client and provides methods to send prompts and receive completions.
Environment variables are set internally to support Azure authentication.
"""
import os
import openai

from .gen_model import GenerativeModel

class AzureOpenAIGM(GenerativeModel):
    """
    Azure-hosted OpenAI language model integration for the EazyML GenAI framework.

    This class initializes an Azure OpenAI client and provides methods to send prompts and receive completions.
    Environment variables are set internally to support Azure authentication.

    Attributes:
        client (openai.AzureOpenAI): An initialized Azure OpenAI client instance.

    Args:
        model_name (str, optional): The name of the deployed model (e.g., "gpt-4o"). Defaults to "gpt-4o".
        azure_endpoint (str): Azure OpenAI endpoint URL.
        api_key (str): API key for authenticating with Azure OpenAI.
        api_version (str): API version (e.g., "2024-02-15-preview").
        **kwargs: Additional keyword arguments for the base `LanguageModel`.

    Raises:
        Exception: If any of the required parameters (`azure_endpoint`, `api_key`, or `api_version`) are missing.
    """
    def __init__(self, 
                 model_name = "gpt-4o",
                 azure_endpoint = None,
                 api_key = None,
                 api_version = None,
                 **kwargs):
        """
        Initializes the AzureOpenAI class and sets up authentication and API configuration.

        Args:
            model_name (str): Name of the deployed model on Azure.
            azure_endpoint (str): Azure OpenAI endpoint.
            api_key (str): API key for Azure OpenAI.
            api_version (str): API version string.
            **kwargs: Additional keyword arguments passed to the base class.

        Raises:
            Exception: If required values are missing for endpoint, API key, or API version.
        """
        if azure_endpoint :
            os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
        else :
            raise Exception("No AZURE_OPENAI_ENDPOINT found")
        if api_key :
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
        else :
            raise Exception("No AZURE_OPENAI_API_KEY found")
        if api_version :
            os.environ["DEPLOYMENT_VERSION"] = api_version
        else :
            raise Exception("No DEPLOYMENT_VERSION found")
        
        self.client = openai.AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version = os.getenv("DEPLOYMENT_VERSION"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        # pass provider name to super class
        super().__init__(
                provider = 'azure_openai',
                model_name = model_name
            )

    def predict(self, messages, **kwargs):
        """
        Sends a chat prompt to the Azure OpenAI model and returns the generated response.

        Args:
            messages (list): List of messages in OpenAI-compatible chat format.
            **kwargs: Optional arguments:
                - tools (list | dict, optional): Tool/function specs for model-assisted completions.

        Returns:
            object: The response object returned from Azure OpenAI.
        """
        # set tools
        tools = None
        if 'tools' in kwargs :
            if isinstance(kwargs['tools'], list) :
                tools = kwargs['tools']
            elif isinstance(kwargs['tools'], dict) :
                tools = [kwargs['tools']]
            
        response = self.client.chat.completions.create(
                model = self.model_name,
                messages = messages,
                tools = tools
        )
        return response


