"""
Wrapper class for Anthropic's Claude language models within the EazyML GenAI framework.

This class handles authentication, message formatting, and response parsing for 
interacting with Anthropic's language models such as Claude 3.5 Sonnet.
"""
import os
import anthropic

from .gen_model import GenerativeModel


class AnthropicGM(GenerativeModel):
    """
    Wrapper class for Anthropic's Claude language models within the EazyML GenAI framework.

    This class handles authentication, message formatting, and response parsing for 
    interacting with Anthropic's language models such as Claude 3.5 Sonnet.

    Attributes:
        client (`anthropic.Anthropic`): Initialized Anthropic API client.
        model_name (`str`): The name of the Claude model used.

    Args:
        - **model_name** (`str, optional`): Name of the Claude model to use. Defaults to "claude-3-5-sonnet-20241022".
        - **api_key** (`str, optional`): Anthropic API key. If not provided, will use `ANTHROPIC_API_KEY` from environment.
        - **kwargs**: Additional keyword arguments passed to the base class.
    
    Raises:
        Exception: If no API key is provided or found in environment.
    """ 

    def __init__(self,
                 model_name="claude-3-5-sonnet-20241022",
                 api_key=None,
                 **kwargs):
        """
        Initializes the Anthropic client and sets up authentication.

        Args:
            model_name (str): The Claude model version to use.
            api_key (str, optional): API key for Anthropic. Uses environment variable if not provided.
            **kwargs: Additional keyword arguments for base LanguageModel class.

        Raises:
            Exception: If no API key is found or set.
        """
        # set anthropic api_key
        if api_key :
            os.environ['ANTHROPIC_API_KEY'] = api_key
        elif os.environ.get('ANTHROPIC_API_KEY') is None:
            raise Exception("No ANTHROPIC_API_KEY found")
        else :
            api_key = os.environ['ANTHROPIC_API_KEY']
        
        # initialize anthropic client
        self.client = anthropic.Anthropic()

        super().__init__(
                provider = 'anthropic',
                model_name = model_name
            )
        
    def predict(self, messages, **kwargs):
        """
        Sends a message prompt to the Anthropic model and returns the generated response.

        Args:
            - **messages** (`list`): List of message dicts in Claude-compatible format.
            - **kwargs**: Optional arguments:
                - system (str): System message providing global instructions.
                - tools (list): List of tools to provide for tool-augmented responses.
                - tool_choice (dict): Specifies which tool to use or auto.

        Returns:
            object: The full response object from the Anthropic client.
        """
        # system message
        system = ""
        if 'system' in kwargs :
            system = kwargs['system']

        # set tools
        tools = []
        if 'tools' in kwargs :
            tools = kwargs['tools']

        # set tool choice
        tool_choice = {"type": "auto"}
        if 'tool_choice' in kwargs :
            tool_choice = kwargs['tool_choice']
        
        response = self.client.messages.create(
                model = self.model_name,
                max_tokens = 1024,
                temperature = 0,
                system = system,
                messages = messages,
                tools = tools,
                tool_choice = tool_choice
        )
        return response
    
    def parse_content(self, content):
        pass
    
    def parse(self, response):
        final_response = []
        input_token_count = 0
        output_token_count = 0
        for content in response.content :
            self.parse_content(content)
        return final_response, input_token_count, output_token_count
