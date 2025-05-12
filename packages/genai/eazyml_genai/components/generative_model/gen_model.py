from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import textwrap


class GenerativeModelProvider(Enum):
    GOOGLE = 'google'
    OPENAI = 'openai'
    OPENAI_AZURE = 'openai_azure'
    VERTEXAI = 'vertexai'
    ANTRHROPIC = 'anthropic'

class GenerativeModel(ABC):

    def __init__(self, **kwargs):
        self.__provider = kwargs.get('provider', None)
        self.__model = kwargs.get('model', None)
        self.__client = kwargs.get('client', None)

    @property
    def provider(self):
        return self.__provider
    
    @property
    def model(self):
        return self.__model
    
    @property
    def client(self):
        return self.__client
    

    def get_extra_context_info(self, payloads):
        extra_context = []
        for payload in payloads:
            if len(payload.get("path", [])) > 0:
                type = payload['type']
                extra_context.extend([{'type': type, 'path': path} for path in payload.get('path', [])])
        return extra_context


    def get_context_info(self, payloads, doc_id=1):
        context = ""
        for payload in payloads:
            doc_text = f"""
            Title:
            {payload.get('title', '')}
            Content:
            {payload.get('content', '')}
            """
            context += textwrap.dedent(doc_text).strip() + "\n\n"
            doc_id += 1
        return context, doc_id
    
    @staticmethod
    def get_table_response(kwargs):
        return {
            'found': kwargs.get('found', False),
            'answer': kwargs.get('answer'),
            'reason': kwargs.get('reason')  
        }
    
    
    @staticmethod 
    def get_image_response(kwargs):
        return {
            'found': kwargs.get('found', False),
            'answer': kwargs.get('answer'),
            'reason': kwargs.get('reason')
        }
