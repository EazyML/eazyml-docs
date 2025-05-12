
from eazyml_genai.components.document_cleaner.document_cleaner import DocumentCleaner


class PDFCleaner(DocumentCleaner):
    
    def __init__(self, file_path: str = None, page_no = None):
        super().__init__(file_path=file_path, page_no=page_no)
        
        