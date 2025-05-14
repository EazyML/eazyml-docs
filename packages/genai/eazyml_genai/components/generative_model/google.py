"""
Google Generative AI (Gemini) integration for the EazyML GenAI framework.

This class wraps the Gemini API using the `google.generativeai` library,
handling model configuration, prompt generation, token counting, and output parsing.
"""
import os
import ast
import asyncio
from pyexpat import model
import textwrap
from google import genai
from ...prompts import PromptTemplate
from google.genai import types
from PIL import (
    Image,
    UnidentifiedImageError
)
from .gen_model import (
    GenerativeModel,
    GenerativeModelProvider
)

from ...license import (
        validate_license
)

get_table_response_fd = {
    "name": "get_table_response",
    "description": "Extract the whole table data as a markdown format. Don't miss any information.",
    "parameters": {
        "type": "object",
        "properties": {
            "found": {
                "type": "boolean",
                "description": "boolean indicating if the question can be answered using the given image",
            },
            "answer": {
                "type": "string",
                "description": "the extracted answer if found is true, otherwise it should be None",
            },
            "description": {
                "type": "string",
                "description": "Extract the whole table data as a markdown format. Ensure that each cell is properly aligned within its column using spaces."
            }
        },
        "required": ["found", "answer", "description"],
    },
}

get_image_response_fd = {
    "name": "get_image_response",
    "description": "Extract content based on given image",
    "parameters": {
        "type": "object",
        "properties": {
            "found": {
                "type": "boolean",
                "description": "boolean indicating if the question can be answered using the given image",
            },
            "answer": {
                "type": "string",
                "description": "the extracted answer if found is true, otherwise it should be None",
            },
            "description": {
                "type": "string",
                "description": "extracted information from given image"
            }
        },
        "required": ["found", "answer", "description"],
    },
}


class GoogleGM(GenerativeModel):
    """
    Google Generative AI (Gemini) integration for the EazyML GenAI framework.
    This class wraps the Gemini API using the `google.generativeai` library,
    handling model configuration, prompt generation, token counting, and output parsing.

    Args:
        **model_name** (`str`, `optional`): The name of the Gemini model to use. Defaults to "gemini-2.0-flash".
        **api_key** (`str`, `optional`): Google API key for authentication. If not provided, it falls back to the 'GEMINI_API_KEY' environment variable.
        **kwargs**: Optional parameters for custom safety settings and generation configuration.
    
    Example:
        .. code-block:: python

            # initialize google generative model
            google_gm = GoogleGM(model="gemini-2.0-flash",
                     api_key=os.getenv('GEMINI_API_KEY'))
        
            # response from generative model given question and retrieved documents
            response, input_tokens, output_tokens = google_gm.predict(question=question,
                                        payloads=payloads,
                                        show_token_details=True
                                        )
            
            # parse google response to simple text format
            parsed_response = google_gm.parse(response=response)
    """

    def __init__(self, model="gemini-2.0-flash-lite", api_key=None, **kwargs):
        # set GEMINI API KEY
        if api_key == None and 'GEMINI_API_KEY' in os.environ:
            api_key = os.getenv('GEMINI_API_KEY')
        # show number of input and output tokens
        self.show_token_details=kwargs.get('show_token_details', False)
        # Setting temperature to 0 for deterministic output
        self.temperature=kwargs.get('temperature', 0)   
        # Consider the top 40 most likely tokens
        self.top_k=kwargs.get('top_k', 40)
        # Example value for top_p (range 0.0 - 1.0)
        self.top_p=kwargs.get('top_p', 1.0)         
        # Generate only one response
        self.candidate_count=kwargs.get('candidate_count', 1)
                
        super().__init__(
                provider = GenerativeModelProvider.GOOGLE,
                model = model,
                client = genai.Client(api_key=api_key)
            )
    

    async def extract_image_data(self, question, extra_info, **kwargs):
        try:
            get_table_response_fd = kwargs.get('table_fd')
            get_image_response_fd = kwargs.get('image_fd')
            # Configure function calling mode
            tools = types.Tool(function_declarations=[get_table_response_fd, get_image_response_fd])
            # if mode is "ANY", then allowed_function_names =["get_table_response", "get_image_response"]
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="AUTO"
                )
            )
            # Create the generation config
            config = types.GenerateContentConfig(
                temperature=0,    # Setting temperature to 0 for deterministic output
                top_k=40,          # Consider the top 40 most likely tokens
                top_p=1.0,         # Example value for top_p (range 0.0 - 1.0)
                candidate_count=1, # Generate only one response
                tools=[tools],
                tool_config=tool_config,
            )

            path = extra_info['path']
            prompt = PromptTemplate.from_template(textwrap.dedent(
                """
                Answer the below question using given table or formula or normal image.
                Create a JSON structure with the keys `found`, `answer`, and `description`.
                The value of `found` should be a boolean indicating the presence of
                the answer for given question in the image. `answer`
                (the extracted answer if found is true, otherwise it should
                be None). The `description` key should contain the information
                extracted from the image.
                
                Question:
                {question}
                """)).invoke({'question': question})
            with Image.open(path) as img:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt, img],
                    config=config,
                )
                input_token_count = response.usage_metadata.prompt_token_count
                output_token_count = response.usage_metadata.candidates_token_count
                # Check if the response contains a function call
                if response.candidates[0].content.parts[0].function_call:
                    function_call = response.candidates[0].content.parts[0].function_call
                    # In a real app, you would call your function here:
                    # Create the FunctionResponse
                    function_response = types.FunctionResponse(
                        name=function_call.name,
                        response=function_call.args
                    )
                    result = types.Part(function_response=function_response)
                else:
                    result = response.text
                return result, input_token_count, output_token_count
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at path: {path}")
        except UnidentifiedImageError:
            raise ValueError(f"Invalid image file or unsupported format: {path}")
        except Exception as e:
            raise RuntimeError(f"{str(e)}")

    async def get_extra_infos(self, question, extra_infos, **kwargs):
        extra_tasks = []
        for extra_info in extra_infos:
            extra_tasks.append((question, extra_info))
                
        results = await asyncio.gather(
                *[self.extract_image_data(ques, info, **kwargs)
                for ques, info in extra_tasks]
        )
        return results

    def predict(self, question, payloads, **kwargs):
        """
        Generates a response from the Gemini model based on the provided prompt.

        Args:
            **prompt** (`str`): Input text or prompt for the model.

            **kwargs** (`dict`):

                    - `safety_settings` (`dict`, `optional`): Dictionary defining harm category thresholds.
                    - `generation_config` (`GenerationConfig`, `optional`): Configuration object for generation behavior.
                    - `tool_config` (`dict`, `optional`): Tool configuration for function calling.
                    - `tools` (`list`, `optional`): Tools to be used with function calling.

        Returns:
            `response`: The response object containing generated candidates.
        """
        # show number of input and output tokens
        self.show_token_details=kwargs.get('show_token_details', self.show_token_details)
        # Setting temperature to 0 for deterministic output
        self.temperature=kwargs.get('temperature', self.temperature)   
        # Consider the top 40 most likely tokens
        self.top_k=kwargs.get('top_k', self.top_k)
        # Example value for top_p (range 0.0 - 1.0)
        self.top_p=kwargs.get('top_p', self.top_p)         
        # Generate only one response
        self.candidate_count=kwargs.get('candidate_count', self.candidate_count)
        # count total input and output tokens
        total_input_token = 0
        total_output_token = 0
        # append all the prompt from image, table or retrieved documents to prompts
        prompts = []
        prompts.append(
            f"""
            Answer the below question using documents below. Just use those document where you find the answer and don't miss any information.
            Don't answer in dictionary/json format.

            Question:
            {question}
            """
        )
        doc_id = 1
        
        # here we extract all the image path and get description of those
        # images, answer for given question based on given images.
        # Finally we append those extracted content either in form of 
        # FunctionResponse or just text to our main prompts.
        extra_context = self.get_extra_context_info(payloads=payloads)
        img_responses = asyncio.run(self.get_extra_infos(
                                question=question,
                                extra_infos=extra_context,
                                table_fd=get_table_response_fd,
                                image_fd=get_image_response_fd
                                ))
        for img_result, input_token, output_token in img_responses:
            prompts.append(img_result)
            total_input_token += input_token
            total_output_token += output_token
            doc_id += 1
        
        # construct our context based on retrieved documents and 
        # append it to prompts
        context, doc_id = self.get_context_info(payloads, doc_id=doc_id)
        prompts.append(context)

        config = types.GenerateContentConfig(
                temperature=0,    # Setting temperature to 0 for deterministic output
                top_k=40,          # Consider the top 40 most likely tokens
                top_p=1.0,         # Example value for top_p (range 0.0 - 1.0)
                candidate_count=1 # Generate only one response
            )
        # Finally, generate response for overall prompts.
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompts,
            config=config,
        )
        # also, calculate count overall token for input and output
        total_input_token += response.usage_metadata.prompt_token_count
        total_output_token += response.usage_metadata.candidates_token_count
        if self.show_token_details :
            return response, total_input_token, total_output_token
        else :
            return response

    def parse(self, response):
        """
        Extracts the text content from the first candidate and part of the response.

        Args:
            **response**: The response object, assumed to have a structure containing candidates, content, and parts. The exact type of 'response' is not specified, but it should behave like a nested list/object as shown in the return description.


        Returns:
            (`str`): The text content located at response.candidates[0].content.parts[0].text. Returns the extracted text as a string.
        """
        return response.candidates[0].content.parts[0].text

    def count_tokens(self, prompt):
        """
        Counts the number of tokens in the given prompt.

        Args:
            prompt (str): The input prompt string.

        Returns:
            dict: Token usage details.
        """
        response = self.model.count_tokens(prompt)
        return response
        

