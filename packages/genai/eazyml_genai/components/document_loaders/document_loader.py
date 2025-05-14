from abc import ABC, abstractmethod
from enum import Enum
from torchvision.ops import nms
from typing import List, Tuple, Union
from PIL import Image
from eazyml_genai.components.document_loaders.block import Block
from eazyml_genai.globals.settings import Settings

from PIL import Image
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

import warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")


class DocumentType(Enum):
    PDF_DOCUMENT = 'pdf_document'
    DOCS_DOCUMENT = 'docs_document'


class LayoutLabel(Enum):
    TITLE = 'title'
    TEXT = 'plain text'
    ABANDON = 'abandon'
    FIGURE = 'figure'
    FIGURE_CAPTION = 'figure_caption'
    TABLE = 'table'
    TABLE_CAPTION = 'table_caption'
    TABLE_FOOTNOTE = 'table_footnote'
    ISOLATE_FORMULA = 'isolate_formula'
    FORMULA_CAPTION = 'formula_caption'


class DocumentKeys(Enum):
    TYPE = 'type'
    TITLE = 'title'
    CONTENT = 'content'
    PATH = 'path'
    META = 'meta'


class Document(ABC):
    
    def __init__(self, **kwargs):
        self.type = kwargs.get(DocumentKeys.TYPE.value, None)
        self.title = kwargs.get(DocumentKeys.TITLE.value, None)
        self.content = kwargs.get(DocumentKeys.CONTENT.value, None)
        self.path = kwargs.get(DocumentKeys.PATH.value, [])
        self.meta = kwargs.get(DocumentKeys.META.value, {})
    
    @staticmethod
    def using_blocks(blocks: List[Block]):
        documents: List[Document] = list()
        for block in blocks:
            block_dict = block.to_dict()
            block_dict['meta'] = {
                'file_path': block_dict['file_path'],
                'page_no': block_dict['page_no'],
                'block_no': block_dict['block_no'],
                'bbox': block_dict['bbox']
            }
            document = Document(**block_dict)
            documents.append(document)
        return documents

    def __str__(self):
        return (f"{self.__class__.__name__}("
                f"{DocumentKeys.TYPE.value}: {self.type},"
                f"{DocumentKeys.TITLE.value}: {self.title},"
                f"{DocumentKeys.CONTENT.value}: {self.content},"
                f"{DocumentKeys.PATH.value}: {self.path},"
                f"{DocumentKeys.META.value}: {self.meta},"
                f")"
                )


    def __repr__(self) -> str:
        return self.__str__()


    def to_dict(self):
        return {
            DocumentKeys.TYPE.value : self.type,
            DocumentKeys.TITLE.value : self.title,
            DocumentKeys.CONTENT.value : self.content,
            DocumentKeys.PATH.value: self.path,
            DocumentKeys.META.value : self.meta
        }
    

layout_label_dict = {
        0: LayoutLabel.TITLE,
        1: LayoutLabel.TEXT,
        2: LayoutLabel.ABANDON,
        3: LayoutLabel.FIGURE,
        4: LayoutLabel.FIGURE_CAPTION,
        5: LayoutLabel.TABLE,
        6: LayoutLabel.TABLE_CAPTION,
        7: LayoutLabel.TABLE_FOOTNOTE,
        8: LayoutLabel.ISOLATE_FORMULA,
        9: LayoutLabel.FORMULA_CAPTION
    }


class DocumentLoader(ABC):
    
    blocks: List[Block] = list()

    def __init__(self,
                 type: Union[str, DocumentType],
                 max_chunk_words: int=500):
        self.__type = type
        self.max_chunk_words = max_chunk_words
        
        # Replace with the actual repository ID
        repo_id = "eazymlshubham/DocLayoutYOLO"
        # Replace with the actual .pt filename
        filename = "doclayout_yolo_docstructbench_imgsz1024.pt"      

        # Download the model file from Hugging Face Hub
        # This will download the file to a local cache directory and return the local path
        local_file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")

        # Load the pre-trained model
        # The model is loaded from the local file path obtained from hf_hub_download
        self.model = YOLOv10(local_file_path)

    @abstractmethod
    def load(self, file):
        pass


    @property
    def type(self):
        return self.__type


    def get_layout_bbox(self,
                        image_path,
                        image_size=1024) -> Tuple[List[List[float]], List[LayoutLabel]]:
        
        # Perform prediction
        det_res = self.model.predict(
            image_path,                       # Image to predict
            imgsz=image_size,                       # Prediction image size
            device="cpu",                     # Device to use (e.g., 'cuda:0' or 'cpu')
            verbose=False
        )

        pred_boxes = det_res[0].boxes
        scores, labels, boxes = pred_boxes.data[:, 4], pred_boxes.data[:, 5], pred_boxes.data[:, :4]
        # filter bounding boxes using non-max suppression with iou threshold = 0.005
        keep = nms(boxes=boxes, scores=scores, iou_threshold=0.005)
        scores, labels, boxes = scores[keep], labels[keep], boxes[keep]
        refined_labels, refined_boxes, refined_scores = [], [], []
        for score, label, box in zip(scores, labels, boxes):
            label_name = layout_label_dict[int(label.item())]
            # if label_name == "Picture":  # This is considered an image
            # adding 5% from left and right side only, so it has better grasp of image or table
            box = box.tolist()
            box = [round(i, 2) for i in box]
            if label_name in {LayoutLabel.TABLE, LayoutLabel.FIGURE, LayoutLabel.ISOLATE_FORMULA}:
                box = [box[0]-5, box[1]-5, box[2]+25, box[3]+5]
            refined_labels.append(label_name)
            refined_boxes.append(box)
            refined_scores.append(score)
        return refined_boxes, refined_labels, refined_scores


    def extract_layout_info(self, page):
        # Get the page's bounding box (dimensions in points)
        page_rect = page.rect

        # Extract width and height from the rectangle
        page_width = page_rect.width
        page_height = page_rect.height
        
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        image_width, image_height = image.size
        
        # get temporary image path based on file name and page number
        image_path = Settings.get_page_image_path(self.file_path, self.page_no)
        # save image to temporary path
        image.save(image_path)
        layout_bboxes, layout_labels, layout_scores = self.get_layout_bbox(image_path=image_path)
        
        layout_infos = []
        table_tasks =[]
        for i, (bbox, layout_label) in enumerate(zip(layout_bboxes, layout_labels)):
            layout_info = dict()
            if layout_label in {LayoutLabel.TABLE, LayoutLabel.FIGURE, LayoutLabel.ISOLATE_FORMULA}:
                layout_item_path = ''
                if layout_label == LayoutLabel.ISOLATE_FORMULA:
                    layout_image = image.crop([bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5])
                elif layout_label == LayoutLabel.FIGURE:
                    layout_image = image.crop([bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5])
                elif layout_label == LayoutLabel.TABLE:
                    layout_image = image.crop([bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5])
                layout_img_path = Settings.get_page_layout_path(
                                file_path=self.file_path,
                                page_no=self.page_no,
                                layout_name=f"{layout_label.value}{i}",
                                extension='.png'
                            )
                layout_image.save(layout_img_path)
                layout_info['path'] = layout_img_path
            scale_width = page_width/image_width
            scale_height = page_height/image_height
            layout_info['bbox'] = [round(bbox[0]*scale_width, 2),
                                  round(bbox[1]*scale_height, 2),
                                  round(bbox[2]*scale_width, 2),
                                  round(bbox[3]*scale_height, 2)]
            layout_info['score'] = layout_scores[i]
            layout_info['label'] = layout_label
            layout_infos.append(layout_info)
        
        return layout_infos

    def convert_blocks_to_document_format(self, blocks: List[Block]):
        documents = Document.using_blocks(blocks=blocks)
        return [doc.to_dict() for doc in documents]


