from eazyml_genai.prompts.base import (
    BasePromptTemplate,
    PromptTemplateFormat,
    get_template_variables
)
from typing import Any
from typing_extensions import Self


class PromptTemplate(BasePromptTemplate):
    
    def __init__(self, input_variables, template, template_format, **kwargs):
        super().__init__(input_variables, template, template_format)
        
    
    @classmethod
    def from_template(
        cls,
        template: str,
        template_format: PromptTemplateFormat = "f-string",
        **kwargs: Any,
        ) -> Self:
        input_variables = get_template_variables(template, template_format)
        return cls(
            input_variables=input_variables,
            template=template,
            template_format=template_format,
            **kwargs,
        )
    
    