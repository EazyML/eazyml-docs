
import fitz
from ...license import (
        validate_license
)
from tqdm import tqdm
from typing import Union, List
from .block import Block, BlockSpan, BlockType
from ..document_cleaner.pdf_cleaner import PDFCleaner
from .document_loader import (
    DocumentLoader,
    DocumentType
)
import nest_asyncio
nest_asyncio.apply()

class PDFLoader(DocumentLoader):
    """
    A class to load and process PDF documents.

    Inherits from `DocumentLoader` and specializes in handling PDF files.
    It extracts text blocks from PDF pages, cleans them, and chunks them into smaller segments.
    
    Args:
        **max_chunk_words** (`int`, `optional`): The maximum number of words per text chunk. Defaults to 500.
    
    Example:
        .. code-block:: python

            from eazyml_genai.components import PDFLoader

            
            # Initialize the PDFLoader and mention max_chunk_words
            pdf_loader = PDFLoader(max_chunk_words=800)
            
            # Loads pdf as json document which has keys such as title,
            # content, path for images of formula, table and pictures
            # from pdf and meta information such as page number,
            # paragraph and bounding boxes information.
            documents = pdf_loader.load(file_path='YOUR FILE PATH')
    """
    
    
    def __init__(self, max_chunk_words:int=500):
        super().__init__(type=DocumentType.PDF_DOCUMENT, max_chunk_words=max_chunk_words)
    
    
    def process_page(self, page):
        """
        Extracts raw text blocks from a single PDF page.

        Args:
            page (fitz.Page): The PyMuPDF Page object to process.

        Returns:
            List[Block]: A list of Block objects, each representing a text block found on the page.
        """
        blocks = page.get_textpage().extractDICT()['blocks']
        block_objs: List[Block] = list()
        for block in blocks:
            # define block type
            block_dict = {
                -1: None,
                0: BlockType.TEXT,
                1: BlockType.IMAGE,
                2: BlockType.TABLE
            }
            if isinstance(block.get('type', -1), int):
                type = block_dict[block.get('type', -1)]
            
            if type == BlockType.TEXT:
                for line_no, line in enumerate(block["lines"], start=1):
                    for span in line["spans"]:
                        block_span = BlockSpan(**span)
                        block_obj = Block(
                                        type=BlockType.TEXT, 
                                        page_no=self.page_no,
                                        file_path=self.file_path,
                                        number=block.get('number', None),
                                        lines=line_no,
                                        spans=block_span,
                                    )
                        block_objs.append(block_obj)
        return block_objs


    def load_page(self, page, pdf_cleaner: PDFCleaner) -> List[Block]:
        """
        Loads and cleans the text blocks from a single PDF page.

        Args:
            page (fitz.Page): The PyMuPDF Page object to process.
            pdf_cleaner (PDFCleaner): An instance of the PDFCleaner class for cleaning operations.

        Returns:
            List[Block]: A list of cleaned Block objects extracted from the page.
        """
        layout_infos = self.extract_layout_info(page)
        block_objs = self.process_page(page)
            
        block_objs = pdf_cleaner.clean_block_objs(layout_infos, block_objs)
        return block_objs

    def load(self, file_path: str, pages: Union[int, list, str, None] = None):
        """
        Loads content from a PDF file, optionally for a specific page, cleans it, chunks it, and converts it into a list of document dictionaries.

        Args:
            - **file_path** (`str`): The path to the PDF file to load.
            - **pages** (`int`, `list`): The specific page number to load.

        Returns:
            (`List[dict]`): A list of dictionaries, where each dictionary represents a chunk of text from the PDF. Each dictionary will typically contain keys like 'content', 'metadata' (including page number, file path, etc.).

        Raises:
            Exception: If `pages` is less than 1.
        """
        self.file_path = file_path
        all_blocks: List[Block] = list()
        
        # Open the PDF
        doc = fitz.open(self.file_path)
        selected_pages = []
        if pages :
            try:
                if isinstance(pages, int):
                    selected_pages.append(pages)
                elif isinstance(pages, list):
                    for page_no in pages:
                        if isinstance(page_no, int):
                            selected_pages.append(page_no)
                        elif isinstance(page_no, str):
                            selected_pages.append(int(page_no))
                elif isinstance(pages, str):
                    if '-' in pages:
                        start_page, end_page = [int(page) for page in pages.split('-')]
                        selected_pages.extend(list(range(start_page, end_page+1)))
                    else:
                        selected_pages.append(int(pages))
            except Exception as e:
                raise Exception('provided pages format is not correct,' +
                                'either provide as list of pages or page number.' +
                                str(e))        
        else :
            selected_pages = list(range(1, len(doc)+1))
            
        pdf_cleaner = PDFCleaner(file_path=self.file_path)
        for page_no in tqdm(selected_pages, desc="Reading pages", unit="page"):
            self.page_no = page_no
            page = doc[page_no - 1]
            pdf_cleaner = PDFCleaner(file_path=self.file_path,
                                        page_no=self.page_no)
            page_blocks = self.load_page(page, pdf_cleaner)
    
            if len(page_blocks) > 0 :
                first_block = page_blocks[0]
                if len(all_blocks) > 0 and first_block.title == None:
                    for block in all_blocks[::-1]:
                        if block.type == BlockType.TEXT:
                            block.update_block(first_block)
                            break
                    page_blocks.remove(first_block)
            all_blocks.extend(page_blocks)
        
        # strip or chunk the content of block to max size of max_chunk_words
        chunk_blocks = pdf_cleaner.chunk_blocks(blocks=all_blocks,
                                                max_words=self.max_chunk_words)
        
        # convert list of blocks into list of dictionary document.
        documents = self.convert_blocks_to_document_format(chunk_blocks)
        return documents
