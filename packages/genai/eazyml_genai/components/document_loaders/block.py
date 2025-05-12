from turtle import update
from numpy import isin
from typing import Any, Set, List, Union
from collections.abc import Iterable
from enum import Enum

class BlockType(Enum):
    TEXT = 'text'
    IMAGE = 'image'
    TABLE = 'table'

class BlockFont:
    
    def __init__(self, name: str, size: float):
        names = name.split('-')
        if len(names) > 1:
            self.__name = names[0].strip()
            self.__type = names[1].strip()
        elif len(names) == 1:
            self.__name = names[0].strip()
            self.__type = None
        self.__size = round(size, 2)
        
    @property
    def name(self):
        return self.__name
    
    @property
    def type(self):
        return self.__type
    
    @property
    def size(self):
        return self.__size
    

    @staticmethod
    def update_fonts(list1, list2):
        merged = {}

        # Add all fonts from list1
        for font in list1:
            key = (font.name, font.type, font.size)
            merged[key] = BlockFont('-'.join([font.name, font.type]), font.size)

        # Merge fonts from list2
        for font in list2:
            key = (font.name, font.type, font.size)
            if key not in merged:
                merged[key] = BlockFont('-'.join([font.name, font.type]), font.size)

        return list(merged.values())


    def __eq__(self, other):
        return (isinstance(other, BlockFont) and
                self.name == other.name and
                self.type == other.type and
                self.size == other.size)


    def __hash__(self):
        return hash((self.name, self.type, self.size))


    def __str__(self):
        return (f"{self.__class__.__name__}("
                f"name: {self.name},"
                f"type: {self.type},"
                f"size: {self.size})"
                )


    def __repr__(self) -> str:
        return self.__str__()


    def to_dict(self):
        return {
            "name": self.name,
            "type": self.type,
            "size": self.size
        }


class BlockSpan:

    def __init__(self, **span):
        self.__text = span["text"] if span["text"] else ''
        # update bbox of span
        self.update_bbox(span["bbox"])
        self.__count = len(span["text"])
        self.__font: BlockFont = BlockFont(span["font"], span["size"])

    def update_span(self, span):
        if isinstance(span, BlockSpan) and self.font == span.font:
            self.update_bbox(span.bbox)
            self.update_text(span.text)
            self.update_count(span.count)


    def is_similar(self, other):
        return isinstance(other, BlockSpan) and self.font == other.font

    @property
    def text(self):
        return self.__text

    def set_text(self, text: str):
        self.__text = text
        
    def update_text(self, text: str):
        if self.text and text:
            self.set_text(self.text + '\n' + text)
        elif text:
            self.set_text(text)

    @property
    def bbox(self):
        return self.__bbox

    def update_bbox(self, bbox):
        if hasattr(self, f'_{self.__class__.__name__}__bbox') and self.__bbox:
            self.__bbox = [
                round(min(self.__bbox[0], bbox[0]),2),
                round(min(self.__bbox[1], bbox[1]),2),
                round(max(self.__bbox[2], bbox[2]),2),
                round(max(self.__bbox[3], bbox[3]),2),
            ]
        else :
            self.__bbox = [round(bbox[0],2),
                           round(bbox[1],2),
                           round(bbox[2],2),
                           round(bbox[3],2)]

    @property
    def count(self):
        return self.__count

    def set_count(self, count):
        self.__count = count
    
    def update_count(self, count):
        self.set_count(self.count + count)

    @property
    def font(self):
        return self.__font
    
    def __eq__(self, other):
        return isinstance(other, BlockSpan) and self.__text == other.__text


    def __hash__(self):
        return hash((self.__text))


    def __str__(self):
        return (f"{self.__class__.__name__}("
                f"text: {self.__text},"
                f"count: {self.__count},"
                f"bbox: {self.__bbox},"
                f"font: {self.font})"
                )


    def __repr__(self) -> str:
        return self.__str__()


    def to_dict(self):
        return {
            "text": self.__text,
            "count": self.__count,
            "bbox": self.__bbox,
            "font": self.font.to_dict()
        }


class Block:
    
    def __init__(self, **kwargs):
        self.__type = kwargs.get('type')
        self.__block_no = kwargs.get('number')
        self.__page_no = kwargs.get('page_no', None)
        self.__file_path = kwargs.get('file_path', None)
        self.set_content(kwargs.get('content', None))
        self.update_lines(kwargs.get('lines'))
        self.set_path(kwargs.get('path'))
        if self.type == BlockType.TEXT:
            # update block spans
            self.update_block_using_span(kwargs.get('spans'))
        elif self.type == BlockType.IMAGE or self.type == BlockType.TABLE:
            self.update_bbox(kwargs.get('bbox'))
            self.__spans = []
        


    @staticmethod
    def update_block_font(font_list: List[BlockFont], new_font: BlockFont):
        if new_font in font_list:
            return font_list
        else :
            font_list.append(new_font)
    
    @staticmethod
    def update_block_fonts(font_list: List[BlockFont], new_fonts: List[BlockFont]):
        for f in new_fonts:
            Block.update_block_font(font_list=font_list, new_font=f)

    
    def update_block(self, block):
        if not isinstance(block, Block):
            return self
        self.update_title(block.title)
        if len([line for line in block.lines if line not in self.lines]) > 0:
            self.update_content(f"\n{block.content}")
        else :
            self.update_content(block.content)
        self.update_lines(self.lines)
        self.set_path(block.path)
        self.set_title_fonts(BlockFont.update_fonts(self.title_fonts, block.title_fonts))
        self.set_content_fonts(BlockFont.update_fonts(self.content_fonts, block.content_fonts))
        self.update_bbox(block.bbox)
        self.update_block_spans(block.spans)
        return self
    
    def update_block_using_span(self, span: BlockSpan):
        if not isinstance(span, BlockSpan):
            return self
        
        self.set_content(span.text)

        # update bbox
        self.update_bbox(span.bbox)

        # update spans
        self.update_block_spans(span)
        
    
    @property
    def type(self):
        return self.__type
    
    @property
    def bbox(self):
        if hasattr(self, f'_{self.__class__.__name__}__bbox') and self.__bbox:
            return self.__bbox
        else:
            return []
        
    def update_bbox_using_spans(self, spans: List[BlockSpan] = []):
        for span in spans:
            self.update_bbox(span.bbox)
    
    def update_bbox(self, bbox):
        if hasattr(self, f'_{self.__class__.__name__}__bbox') and self.__bbox:
            self.__bbox = [
                round(min(self.__bbox[0], bbox[0]),2),
                round(min(self.__bbox[1], bbox[1]),2),
                round(max(self.__bbox[2], bbox[2]),2),
                round(max(self.__bbox[3], bbox[3]),2),
            ]
        else :
            self.__bbox = [round(bbox[0],2),
                           round(bbox[1],2),
                           round(bbox[2],2),
                           round(bbox[3],2)]


    def update_block_spans(self, span: Union[BlockSpan, List[BlockSpan]]):
        # update block spans
        if (not hasattr(self, f'_{self.__class__.__name__}__spans') or
                self.__spans is None or
                len(self.__spans) == 0):
            self.__spans = []
        if isinstance(span, BlockSpan):
            self.__spans.append(span)
        elif isinstance(span, list):
            self.__spans.extend(span)
        
    
    @property
    def content(self):
        if hasattr(self, f'_{self.__class__.__name__}__content'):
            return self.__content
        else:
            return None

    def set_content(self, content: Union[str, None]):
        self.__content = content
        
    def update_content(self, content: str):
        if self.content and content:
            self.set_content(self.content + content)
        elif content:
            self.set_content(content)

    def update_lines(self, lines: Union[list, int]):
        if not hasattr(self, f'_{self.__class__.__name__}__lines'):
            self.__lines = []
        if lines:
            if isinstance(lines, int):
                if lines not in self.__lines:
                    self.__lines.append(lines)
            elif isinstance(lines, list):
                self.__lines.extend([line for line in lines if line not in self.__lines])
    
    @property
    def lines(self):
        return self.__lines

    @property
    def path(self):
        if hasattr(self, f'_{self.__class__.__name__}__path'):
            return self.__path
        else:
            return []

    def set_path(self, path: Union[str, List, None]):
        if path is None:
            if not hasattr(self, f'_{self.__class__.__name__}__path'):
                self.__path = []
            return
    
        if not (isinstance(path, List) or isinstance(path, str)):
            raise TypeError('Invalid block path type')

        if not hasattr(self, f'_{self.__class__.__name__}__path'):
            self.__path = []

        if isinstance(path, str) and path not in self.__path:
            self.__path.append(path)
            
        elif isinstance(path, list):
            new_items = [item for item in path if item not in self.__path]
            self.__path.extend(new_items)
        

    @property
    def file_path(self):
        if hasattr(self, f'_{self.__class__.__name__}__file_path'):
            return self.__file_path
        else:
            return None
    
    
    @property
    def title(self):
        if hasattr(self, f'_{self.__class__.__name__}__title'):
            return self.__title
        else:
            return None


    def set_title(self, title: Union[str, None]):
        self.__title = title


    def update_title(self, title: Union[str, None]):
        if self.title and title:
            self.set_title(self.title + ' ' + title)
        elif title:
            self.set_title(title)
        
        
    @property
    def combined_title(self):
        if hasattr(self, f'_{self.__class__.__name__}__combined_title'):
            return self.__combined_title
        else:
            return None

    def set_combined_title(self, combined_title: Union[str, None]):
        self.__combined_title = combined_title
        
    def update_combined_title(self, combined_title: Union[str, None]):
        self.set_combined_title(combined_title + '\n' + self.combined_title)
    
    @property
    def title_fonts(self) -> List[BlockFont]:
        if hasattr(self, f'_{self.__class__.__name__}__title_fonts'):
            return self.__title_fonts
        else:
            return []
        

    def set_title_fonts(self, title_fonts):
        self.__title_fonts = title_fonts
    
    def update_title_fonts(self, title_fonts):
        for title_font in title_fonts:
            Block.update_block_font(self.title_fonts, title_font)

    @property
    def combined_title_fonts(self) -> List[BlockFont]:
        if hasattr(self, f'_{self.__class__.__name__}__combined_title_fonts'):
            return self.__combined_title_fonts
        else:
            return []

    def set_combined_title_fonts(self, combined_title_fonts):
        self.__combined_title_fonts = combined_title_fonts
    
    def update_combined_title_fonts(self, combined_title_fonts):
        for combined_title_font in combined_title_fonts:
            Block.update_block_font(self.__combined_title_fonts, combined_title_font)
    
    @property
    def content_fonts(self)-> List[BlockFont]:
        if hasattr(self, f'_{self.__class__.__name__}__content_fonts'):
            return self.__content_fonts
        else:
            return []

    def set_content_fonts(self, content_fonts):
        self.__content_fonts = content_fonts
    
    def update_content_fonts(self, content_fonts):
        for content_font in content_fonts:
            Block.update_block_font(self.content_fonts, content_font)
    
    @property
    def spans(self):
        if hasattr(self, f'_{self.__class__.__name__}__spans'):
            return self.__spans
    
    def set_spans(self, spans: List[BlockSpan]):
        if not isinstance(spans, list):
            raise TypeError('Invalid block lines type')
        self.__spans = spans
    
    @property
    def page_no(self):
        if hasattr(self, f'_{self.__class__.__name__}__page_no'):
            return self.__page_no
        else:
            return None

    @property
    def block_no(self):
        if hasattr(self, f'_{self.__class__.__name__}__block_no'):
            return self.__block_no
        else:
            return None
    
    def __eq__(self, other):
        return (
                isinstance(other, Block) and
                self.type == other.type and
                self.page_no == other.page_no and
                self.block_no is not None and
                other.block_no is not None and
                self.block_no == other.block_no and
                self.title == other.title and
                self.content == other.content and
                self.file_path == other.file_path
        )

    def __str__(self):
        return (f"{self.__class__.__name__}("
                f"type: {self.type},"
                f"combined_title: {self.combined_title},"
                f"title: {self.title},"
                f"content: {self.content},"
                f'bbox: {self.bbox},'
                f'lines: {self.lines},'
                f'combined_title_fonts: {self.combined_title_fonts},'
                f'title_fonts: {self.title_fonts},'
                f'content_fonts: {self.content_fonts},'
                f'file_path: {self.file_path},'
                f'page_no: {self.page_no},'
                f'block_no: {self.block_no},'
                f'spans: {self.spans},'
                f'path: {str(self.path)})'
            )
        
    def __repr__(self) -> str:
        return self.__str__()
        
    def to_dict(self):
        return {
                'type': self.type.value,
                'combined_title': self.combined_title,
                'title': self.title,
                'content': self.content,
                'bbox': self.bbox,
                'lines': self.lines,
                # 'combined_title_fonts': [font.to_dict() for font in self.combined_title_fonts],
                # 'title_fonts': [font.to_dict() for font in self.title_fonts],
                # 'content_fonts': [font.to_dict() for font in self.content_fonts],
                'file_path': str(self.file_path),
                'page_no': self.page_no,
                'block_no': self.block_no,
                'spans': [span.to_dict() for span in self.spans],
                'path': [str(p) for p in self.path],
            }
                
