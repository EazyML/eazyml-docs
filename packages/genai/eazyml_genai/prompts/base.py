from abc import ABC, abstractmethod
from typing import Literal, Dict
from string import Formatter

PromptTemplateFormat = Literal["f-string", "mustache", "jinja2"]

def get_template_variables(template: str, template_format: str) -> list[str]:
    """Get the variables from the template.

    Args:
        template: The template string.
        template_format: The template format. Should be one of "f-string" or "jinja2".

    Returns:
        The variables from the template.

    Raises:
        ValueError: If the template format is not supported.
    """
    if template_format == "f-string":
        input_variables = {
            v for _, v, _, _ in Formatter().parse(template) if v is not None
        }
    else:
        msg = f"Unsupported template format: {template_format}"
        raise ValueError(msg)

    return sorted(input_variables)

class BasePromptTemplate(ABC):
    
    def __init__(self, input_variables, template, template_format: PromptTemplateFormat):
        super().__init__()
        self.input_variables = input_variables
        self.template = template
        self.template_format = template_format
    
    
    def invoke(self, input: Dict[str, str]={}) -> str:
        # 1. Validate input contains all required variables
        missing_vars = [var for var in self.input_variables if var not in input]
        if missing_vars:
            raise ValueError(f"Missing input variables: {missing_vars}")

        # 2. Handle template formatting
        if self.template_format == "f-string":
            try:
                result = self.template.format(**input)
            except KeyError as e:
                raise ValueError(f"Missing variable in input: {e}")
            except Exception as e:
                raise ValueError(f"Error formatting template: {e}")
        else:
            raise ValueError(f"Unsupported template format: {self.template_format}")

        return result
        
    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_variables={self.input_variables}, "
            f"template={repr(self.template)})"
        )

    def __repr__(self) -> str:
        return self.__str__()
        