"""
Wrapper class for Google's VertexAI (Gemini) generative models integrated into the EazyML framework.

This class handles authentication, model initialization, and inference with support for
function calling, generation configuration, and safety settings using the `vertexai` SDK.
"""
import os
import json

import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    ToolConfig
)

from .gen_model import GenerativeModel


class VertexAIGM(GenerativeModel):
    """
    Wrapper class for Google's VertexAI (Gemini) generative models integrated into the EazyML framework.

    This class handles authentication, model initialization, and inference with support for
    function calling, generation configuration, and safety settings using the `vertexai` SDK.

    Attributes:
        safety_settings (dict): Safety filters for content moderation.
        generation_config (GenerationConfig): Generation parameters such as temperature, token limits.
        tool_config (ToolConfig): Configuration for enabling/disabling function calling.
        model (GenerativeModel): An initialized Gemini model instance.

    Args:
        model_name (str, optional): Gemini model variant (e.g., 'gemini-1.5-flash'). Defaults to 'gemini-1.5-flash'.
        application_credentials (str, optional): Path to GCP service account credentials file.
        api_key (str, optional): Gemini API key.
        **kwargs: Additional keyword arguments like `project`, `location`, `safety_settings`, and `generation_config`.

    Raises:
        Exception: If neither `GOOGLE_APPLICATION_CREDENTIALS` nor `GEMINI_API_KEY` is available.
    """
    def __init__(self,
                 model_name="gemini-1.5-flash",
                 application_credentials=None,
                 api_key=None,
                 **kwargs):
        """
        Initializes the VertexAI class by configuring authentication, Vertex AI environment,
        and loading a GenerativeModel with the given parameters.

        This setup supports both service account credentials and Gemini API key authentication.
        It also allows customizing safety settings, generation configuration, and tool configuration.

        Args:
            model_name (str, optional): The Gemini model name to load. 
                Defaults to `"gemini-1.5-flash"`.
            application_credentials (str, optional): Path to a GCP service account JSON key file.
                Used to authenticate via service account.
            api_key (str, optional): Gemini API key. Used if service account credentials are not provided.
            **kwargs:
                project (str, optional): GCP project ID for initializing Vertex AI.
                location (str, optional): Region for Vertex AI (e.g., "us-central1").
                safety_settings (dict, optional): Dictionary of safety filters to override default.
                generation_config (GenerationConfig, optional): Parameters such as temperature,
                    max tokens, top-k, and top-p for generation.
                tool_config (ToolConfig, optional): Function calling mode and settings.

        Raises:
            Exception: If neither `application_credentials` nor `api_key` is provided and
                no corresponding environment variables are set.
        """
        # set google application credentials path which includes private key or api_key
        if application_credentials :
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = application_credentials
        elif api_key :
            os.environ['GEMINI_API_KEY'] = api_key
        elif os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') is None and\
                os.environ.get('GEMINI_API_KEY') is None :
            raise Exception("No GOOGLE_APPLICATION_CREDENTIALS or GEMINI_API_KEY found")
        
        # initialize vertexai 
        if 'project' in kwargs and 'location' in kwargs :
            vertexai.init(project="gcp-vpcx-acl", location="us-central1")
        elif 'project' in kwargs or 'location' in kwargs :
            if 'project' in kwargs :
                vertexai.init(project="gcp-vpcx-acl")
            else :
                vertexai.init(location="us-central1")
        else :
            vertexai.init()

        # set SafetySettings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
        if 'safety_settings' in kwargs :
            safety_settings = kwargs['safety_settings']
        self.safety_settings = safety_settings


        # set GenerationConfig
        generation_config = GenerationConfig(
            temperature=0,
            max_output_tokens=4096,
            top_k=1,
            top_p=1,
            candidate_count=1
        )
        if 'generation_config' in kwargs :
            generation_config = kwargs['generation_config']
        self.generation_config = generation_config

        # set tool config
        tool_config = ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.AUTO,
            )
        )
        self.tool_config = tool_config
        
        # initialize model
        model = GenerativeModel(
                            model_name = model_name,
                            safety_settings = self.safety_settings,
                            generation_config = self.generation_config,
                            tool_config=tool_config
                        )
        self.model = model

        # pass provider name to super class
        super().__init__(
                provider = 'vertex_ai',
                model_name = model_name,
            )
        
    def predict(self, prompt, **kwargs):
        """
        Generates a model response for the given prompt.

        Args:
            prompt (str or list): Input to the model. Can be a string or list of message parts.
            **kwargs:
                safety_settings (dict, optional): Override default safety settings.
                generation_config (GenerationConfig, optional): Override generation parameters.
                tool_config (ToolConfig, optional): Override function calling config.
                tools (list, optional): Tool definitions for function calling.

        Returns:
            GenerativeResponse: Response from the model.
        """
        # set SafetySettings
        if 'safety_settings' in kwargs :
            safety_settings = kwargs['safety_settings']
            self.safety_settings = safety_settings
        
        # set GenerationConfig
        if 'generation_config' in kwargs :
            generation_config = kwargs['generation_config']
            self.generation_config = generation_config

        # function calling tool config
        if 'tool_config' in kwargs :
            tool_config = kwargs['tool_config']
            self.tool_config = tool_config

        # function calling tools
        tools = None
        if 'tools' in kwargs :
            tools = kwargs['tools']

        response = self.model.generate_content(
            prompt,
            safety_settings = self.safety_settings,
            generation_config = self.generation_config,
            tool_config = self.tool_config,
            tools = tools
        )
        return response
    
    def count_tokens(self, prompt):
        """
        Estimates token usage for the given prompt.

        Args:
            prompt (str or list): Input content for which token count is calculated.

        Returns:
            CountTokensResponse: Contains metadata including token count.
        """
        response = self.model.count_tokens(prompt)
        return response
    
    def parse_candidate(self,
                        candidate,
                        repeat_count = 0,
                        input_token_count = 0,
                        output_token_count = 0):
        repeat_count += 1
        if hasattr(candidate, 'finish_reason')  :
            user_prompt = ""
            if candidate.finish_reason.name == "STOP" :
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and len(candidate.content.parts) > 0:
                    candidate_response = None
                    part = candidate.content.parts[0]
                    if hasattr(part, 'function_call') :
                        fc = part.function_call
                        candidate_response = f"{fc.name}({type(fc).to_dict(fc)['args']})"
                        return candidate_response, input_token_count, output_token_count
                    elif hasattr(part, 'text'):
                        user_prompt = part.text
            elif candidate.finish_reason.name == "MALFORMED_FUNCTION_CALL" :
                user_prompt = candidate.finish_message
            elif candidate.finish_reason.name in ['BLOCKLIST', 'FINISH_REASON_UNSPECIFIED', 'MAX_TOKENS', 'OTHER', 'PROHIBITED_CONTENT', 'RECITATION', 'SAFETY', 'SPII'] :
                # candidate don't have parsable finish name
                return None, input_token_count, output_token_count
        return None, input_token_count, output_token_count
        
    
    def parse(self, response):
        """
        Parses the complete model response and extracts final output(s) and token metadata.

        Args:
            response (GenerativeResponse): The full model response to parse.

        Returns:
            tuple: (List of parsed responses, total_input_token_count, total_output_token_count)
        """
        final_response = []
        input_token_count = 0
        output_token_count = 0
        for candidate in response.candidates :
            candidate_response, candidate_input_token, candidate_output_token = self.parse_candidate(candidate)
            final_response.append(candidate_response)
            input_token_count += candidate_input_token
            output_token_count += candidate_output_token
        return final_response, input_token_count, output_token_count
