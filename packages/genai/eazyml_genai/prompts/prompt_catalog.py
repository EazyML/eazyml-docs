import textwrap

EXTRACT_TABLE_PROMPT = textwrap.dedent(
    """
    Extract the table data and output it as a Markdown table.
    Use hyphens (-) to create the header row separator.
    Ensure that each cell is properly aligned within its column using spaces.
    If any cells are empty, represent them with an empty string.
    """
)