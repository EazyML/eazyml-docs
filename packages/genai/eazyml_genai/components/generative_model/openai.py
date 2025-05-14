"""
Wrapper for the OpenAI Chat Completion API using the official `openai` SDK.

This class integrates OpenAI's chat models (like GPT-4o) into the EazyML GenAI framework.
It handles environment-based API key configuration and provides a unified interface
for message-based chat completion.
"""
import os
import json
import base64
from io import BytesIO
import openai
import asyncio
import textwrap
from ...prompts import PromptTemplate

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
    "type": "function",
    "function": {
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
            "additionalProperties": False
        }
    }
}

get_image_response_fd = {
    "type": "function",
    "function": {
        "name": "get_image_response",
        "description": "Answer for given question based on given image",
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
            "additionalProperties": False
        }
    },
}



class OpenAIGM(GenerativeModel):
    """
    Wrapper for the OpenAI Chat Completion API using the official `openai` SDK.
    This class integrates OpenAI's chat models (like GPT-4o) into the EazyML GenAI framework.
    It handles environment-based API key configuration and provides a unified interface
    for message-based chat completion.

    Args:
        **model** (`str`, `optional`): The name of the OpenAI model to use. Defaults to 'gpt-4o'.
        
        **api_key** (`str`, `optional`): OpenAI API key. If not provided, will attempt to read from the 'OPENAI_API_KEY' environment variable.

    Raises:
        Exception: If no API key is provided and the environment variable is not set.
    """

    def __init__(self, model = 'gpt-4.1', api_key=None):
        
        # set anthropic api_key
        if api_key :
            os.environ['OPENAI_API_KEY'] = api_key
        elif os.environ.get('OPENAI_API_KEY') is None:
            raise Exception("No OPENAI_API_KEY found")
        else :
            api_key = os.environ['OPENAI_API_KEY']
        
        # pass provider name to super class
        super().__init__(
                provider = GenerativeModelProvider.OPENAI,
                model = model,
                client = openai.OpenAI(api_key=api_key)
            )


    async def extract_image_data(self, question, extra_info, **kwargs):
        try:
            path = extra_info['path']
            get_table_response_fd = kwargs.get('table_fd')
            get_image_response_fd = kwargs.get('image_fd')
            tools = [get_table_response_fd, get_image_response_fd]
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
                # Convert the image to bytes using BytesIO
                buffered = BytesIO()
                img = img.convert("RGB") # Ensure it's in RGB format for broader compatibility
                img.save(buffered, format="JPEG") # Save as JPEG for base64 encoding
                img_bytes = buffered.getvalue()
                b64_string = base64.b64encode(img_bytes).decode("utf-8")
                image_url = f"data:image/jpeg;base64,{b64_string}"
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]}
                    ],
                    tools=tools,
                    tool_choice="required",  # Let GPT decide if it wants to call the function
                )
                input_token_count = response.usage.prompt_tokens
                output_token_count = response.usage.completion_tokens
                # Check for a function call
                if response.choices[0].message.tool_calls[0].function:
                    function_call = response.choices[0].message.tool_calls[0].function
                    #  In a real app, you would call your function here:
                    result = eval(f"self.{function_call.name}({json.loads(function_call.arguments)})")
                else:
                    result = response.choices[0].message.content
                return result, input_token_count, output_token_count
        except FileNotFoundError:
            raise FileNotFoundError(f"Table image not found at path: {path}")
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

    def predict(self, question, payloads, show_token_details=False):
        """
        Sends a chat completion request to the OpenAI API using the provided messages.

        Args:
            **messages** (`list`): A list of message dicts as per OpenAI's chat format (role/content pairs).
            
            **kwargs** (`dict`):
                    - tools (`list` or `dict`, `optional`): Tool definitions for function calling. Can be a single tool (as dict) or a list of tools.

        Returns:
            `openai.types.chat.ChatCompletion`: The OpenAI API response object.
        """
        total_input_token = 0
        total_output_token = 0
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
        if self.model in {'gpt-4.1'}:
            extra_context = self.get_extra_context_info(payloads=payloads)
            img_responses = asyncio.run(self.get_extra_infos(
                                    question=question,
                                    extra_infos=extra_context,
                                    table_fd=get_table_response_fd,
                                    image_fd=get_image_response_fd
                                    ))
            for img_res, input_token, output_token in img_responses:
                if isinstance(img_res, dict):
                    if img_res.get('found', False):
                        prompts.append(textwrap.dedent(
                            f"""
                            Answer:
                            {img_res.get('answer', '')}
                            Reason:
                            {img_res.get('reason', '')}
                            """).strip())
                elif isinstance(img_res, str):
                    prompts.append(textwrap.dedent(
                        f"""
                        Content:
                        {img_res}
                        """).strip())
                total_input_token += input_token
                total_output_token += output_token
                doc_id += 1
        
        context, doc_id = self.get_context_info(payloads, doc_id=doc_id)
        prompts.append(context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": context},
                ]}
            ],
        )
        
        
        total_input_token += response.usage.prompt_tokens
        total_output_token += response.usage.completion_tokens
        if show_token_details :
            return response, total_input_token, total_output_token
        else :
            return response

    def parse(self, response):
        """
        Extracts the text content from the first candidate and part of the response.

        Args:
            **response** (`openai.types.chat.ChatCompletion`):
                    The response object, assumed to have a structure containing candidates, content, and parts.  The exact type of 'response' is not specified, but it should behave like a nested list/object as shown in the return description.
            
        Returns:
            `str`: The text content located at response.candidates[0].content.parts[0].text. Returns the extracted text as a string.
        """
        return response.choices[0].message.content
