from abc import ABC, abstractmethod
from pydoc import text
from typing import Any, Dict, Union, List
import numpy as np
from eazyml_genai.components.document_loaders.block import Block, BlockType
from eazyml_genai.components.document_loaders.document_loader import (
    LayoutLabel
)
# import nltk
import copy
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List

# Make sure NLTK resources are downloaded
# nltk.download('punkt')

class DocumentCleaner(ABC):
    
    def __init__(self, file_path: str = None, page_no = None):
        self.file_path = file_path
        self.page_no = page_no

    def chunk_paragraph(self, paragraph, max_words=100):
        sentences = sent_tokenize(paragraph)
        chunks = []
        current_chunk = ""
        current_word_count = 0

        for sentence in sentences:
            words_in_sentence = word_tokenize(sentence)
            sentence_word_count = len(words_in_sentence)

            if current_word_count + sentence_word_count <= max_words:
                current_chunk += " " + sentence
                current_word_count += sentence_word_count
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_word_count = sentence_word_count

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


    def chunk_blocks(self, blocks: List[Block], max_words=200):
        chunked_blocks = []
        for block in blocks:
            if block.content:
                chunks = self.chunk_paragraph(block.content, max_words=max_words)
                for chunk in chunks:
                    temp_block = copy.deepcopy(block)
                    temp_block.set_content(chunk)
                    chunked_blocks.append(temp_block)
            else:
                chunked_blocks.append(copy.deepcopy(block))
        return chunked_blocks


    def remove_table_blocks(self,
                            blocks:  List[Block]):
        table_blocks: List[Block] = [block for block in blocks if block.type == BlockType.TABLE]
        text_blocks: List[Block] = [block for block in blocks if block.type == BlockType.TEXT]
        image_blocks: List[Block] = [block for block in blocks if block.type == BlockType.IMAGE]
        for table_block in table_blocks:
            table_bbox = table_block.bbox
            # first remove already text inside formula blocks
            i = 0
            while(i < len(text_blocks)):
                text_block = text_blocks[i]
                text_center_x, text_center_y= self.center_coordinates(text_block.bbox)
                if (table_bbox[0] < text_center_x
                    and table_bbox[1]< text_center_y
                    and table_bbox[2] > text_center_x
                    and table_bbox[3]> text_center_y):
                    if text_block in text_blocks :
                        table_block.update_content(text_block.content)
                        table_block.update_lines(text_block.lines)
                        table_block.update_content_fonts([span.font for span in text_block.spans])
                        text_blocks.remove(text_block)
                        continue
                i += 1
        blocks = text_blocks + table_blocks + image_blocks
        return blocks


    def remove_image_blocks(self,
                            blocks:  List[Block]):
        table_blocks: List[Block] = [block for block in blocks if block.type == BlockType.TABLE]
        text_blocks: List[Block] = [block for block in blocks if block.type == BlockType.TEXT]
        image_blocks: List[Block] = [block for block in blocks if block.type == BlockType.IMAGE]
        for image_block in image_blocks:
            image_bbox = image_block.bbox
            # first remove already text inside formula blocks
            i = 0
            while(i < len(text_blocks)):
                text_block = text_blocks[i]
                text_center_x, text_center_y= self.center_coordinates(text_block.bbox)
                if (image_bbox[0] < text_center_x
                    and image_bbox[1]< text_center_y
                    and image_bbox[2] > text_center_x
                    and image_bbox[3]> text_center_y):
                    if text_block in text_blocks :
                        image_block.update_content(text_block.content)
                        image_block.update_lines(text_block.lines)
                        image_block.update_content_fonts([span.font for span in text_block.spans])
                        text_blocks.remove(text_block)
                        continue
                i += 1
        blocks = text_blocks + table_blocks + image_blocks
        return blocks
    

    def center_coordinates(self, bbox):
        if bbox:
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            return center_x, center_y
        else :
            return None, None


    def find_center_distance(self, bbox1, bbox2):
        center1 = np.array([self.center_coordinates(bbox1)])
        center2 = np.array([self.center_coordinates(bbox2)])
        distance = np.linalg.norm(center1 - center2)
        return distance

    def remove_unnecessary_layout(self, layout_infos: List[Dict],
                      block_objs: List[Block],
                      labels: List[LayoutLabel] = [LayoutLabel.ABANDON]
                      ):
        refined_layout_infos = []
        for layout_info in layout_infos:
            found = False
            if layout_info['label'] in labels:
                i = 0 
                while i < len(block_objs) and not found:
                    block_center = self.center_coordinates(block_objs[i].bbox)
                    layout_bbox = layout_info['bbox']
                    if (block_center[0] > layout_bbox[0] and
                        block_center[0] < layout_bbox[2] and
                        block_center[1] > layout_bbox[1] and
                        block_center[1] < layout_bbox[3]):
                        del block_objs[i]
                        found = True
                    i += 1
            if not found:
                refined_layout_infos.append(layout_info)
        
        return refined_layout_infos, block_objs

    def add_image_table_blocks(self,
                               layout_infos: List[Dict], block_objs: List[Block]):
        # removing earlier image block extracted from pymupdf
        block_objs = [i for i in block_objs if i.type == BlockType.TEXT]
        
        refined_layout_info = list()
        # adding image and table block extracted from layout detection model
        for layout_info in layout_infos:
            if layout_info['label'] in {LayoutLabel.FIGURE, LayoutLabel.TABLE}:
                if layout_info['label'] == LayoutLabel.FIGURE:
                    block_objs.append(Block(**layout_info,
                                            type=BlockType.IMAGE,
                                            file_path=self.file_path,
                                            page_no=self.page_no
                                            ))
                elif layout_info['label'] == LayoutLabel.TABLE:
                    block_objs.append(Block(**layout_info,
                                            # since we started saving table content in format of
                                            # image then no need to read content from table path
                                            # content=get_content_from_path(layout_info['path']),
                                            type=BlockType.TABLE,
                                            file_path=self.file_path,
                                            page_no=self.page_no
                                            ))
            else :
                refined_layout_info.append(layout_info)
        return refined_layout_info, block_objs

    
    def update_formula_blocks(self, layout_infos: List[Dict],
                            block_objs: List[Block],
                            labels: List[LayoutLabel] = [LayoutLabel.ISOLATE_FORMULA],
                            distance_threshold=50):
        refined_layout_infos = []
        for layout_info in layout_infos:
            index = -1
            if layout_info['label'] in labels:
                layout_bbox = layout_info['bbox']
                # first remove already text inside formula blocks
                # text_blocks = [i for i in block_objs if i.type == BlockType.TEXT]
                # for text_block in text_blocks:
                #     text_center_x, text_center_y= self.center_coordinates(text_block.bbox)
                #     if (layout_bbox[0] < text_center_x
                #         and layout_bbox[1]< text_center_y
                #         and layout_bbox[2] > text_center_x
                #         and layout_bbox[3]> text_center_y):
                #         block_objs.remove(text_block)
                
                # find nearest block and add this formula path to it
                max_block_distance = 3000000
                for i, block_obj in enumerate(block_objs):
                    if block_obj.type == BlockType.TEXT and layout_info['bbox'] and block_obj.bbox:
                        layout_block_distance = self.find_center_distance(layout_info['bbox'], block_obj.bbox)
                        if layout_block_distance < max_block_distance and layout_block_distance < distance_threshold:
                            max_block_distance = layout_block_distance
                            index = i
            if index != -1 :
                block_objs[index].set_path([layout_info['path']])
            else :
                refined_layout_infos.append(layout_info)
        return refined_layout_infos, block_objs


    def update_title_blocks(self, layout_infos: List[Dict],
                            block_objs: List[Block],
                            labels: List[LayoutLabel] = [LayoutLabel.TITLE],
                            title_threshold=0.7):
        refined_layout_infos = []
        for layout_info in layout_infos:
            found = False
            if layout_info['label'] in labels and layout_info['score'] > title_threshold:
                layout_bbox = layout_info['bbox']
                # first remove already text inside formula blocks
                text_blocks = [i for i in block_objs if i.type == BlockType.TEXT]
                for text_block in text_blocks:
                    text_center_x, text_center_y= self.center_coordinates(text_block.bbox)
                    if (layout_bbox[0] < text_center_x
                        and layout_bbox[1]< text_center_y
                        and layout_bbox[2] > text_center_x
                        and layout_bbox[3]> text_center_y):
                        if text_block.content :
                            text_block.set_title(text_block.content)
                            text_block.set_title_fonts(text_block.content_fonts)
                            text_block.set_content(None)
                            text_block.set_content_fonts([])
                            
                
            if not found:
                refined_layout_infos.append(layout_info)
        return refined_layout_infos, block_objs

    def update_table_caption(self,
                            layout_infos: List[Dict],
                            block_objs: List[Block],
                            labels: List[LayoutLabel] = [LayoutLabel.FIGURE_CAPTION, LayoutLabel.TABLE_CAPTION],
                            distance_threshold=1000):
        
        # Identify all table blocks and remove those table from from block objs
        table_blocks: List[Block] = list()
        i = 0
        while i < len(block_objs):
            if block_objs[i].type == BlockType.TABLE:
                table_blocks.append(block_objs.pop(i))
                continue
            i += 1
        
        
        for table_block in table_blocks:
            max_block_distance = 3000000
            caption_layout_bbox = None
            index = -1
            for j, layout_info in enumerate(layout_infos):
                if layout_info['label'] in labels and layout_info['bbox']:
                    image_caption_distance = self.find_center_distance(layout_info['bbox'], table_block.bbox)
                    if image_caption_distance < max_block_distance and image_caption_distance < distance_threshold:
                        max_block_distance = image_caption_distance
                        index = j
                        caption_layout_bbox = layout_info['bbox']
            if index != -1:
                del layout_infos[index]
            
            selected_caption_block = None
            i = 0
            while i < len(block_objs) and caption_layout_bbox:
                block_center = self.center_coordinates(block_objs[i].bbox)
                layout_bbox = caption_layout_bbox
                # Figure out if the block satified for being the table caption of footnote condition
                if (block_center[0] > layout_bbox[0] and
                    block_center[0] < layout_bbox[2] and
                    block_center[1] > layout_bbox[1] and
                    block_center[1] < layout_bbox[3]):
                    selected_caption_block = block_objs.pop(i)
                    break
                i += 1
            
            if selected_caption_block:
                table_block.update_bbox(selected_caption_block.bbox)
                table_block.update_title(selected_caption_block.content)
                table_block.update_title_fonts([span.font for span in selected_caption_block.spans])

        block_objs.extend(table_blocks)
        return layout_infos, block_objs

    def update_image_caption(self,
                            layout_infos: List[Dict],
                            block_objs: List[Block],
                            labels: List[LayoutLabel] = [LayoutLabel.FIGURE_CAPTION, LayoutLabel.TABLE_CAPTION],
                            distance_threshold=1000):
        
        # Identify all table blocks and remove those table from from block objs
        image_blocks: List[Block] = list()
        i = 0
        while i < len(block_objs):
            if block_objs[i].type == BlockType.IMAGE:
                image_blocks.append(block_objs.pop(i))
                continue
            i += 1
        
        for image_block in image_blocks:
            max_block_distance = 3000000
            caption_layout_bbox = None
            index = -1
            for j, layout_info in enumerate(layout_infos):
                if layout_info['label'] in labels and layout_info['bbox']:
                    image_caption_distance = self.find_center_distance(layout_info['bbox'], image_block.bbox)
                    if image_caption_distance < max_block_distance and image_caption_distance < distance_threshold:
                        max_block_distance = image_caption_distance
                        index = j
                        caption_layout_bbox = layout_info['bbox']
            if index != -1:
                del layout_infos[index]
            
            selected_caption_block = None
            i = 0
            while i < len(block_objs) and caption_layout_bbox:
                block_center = self.center_coordinates(block_objs[i].bbox)
                layout_bbox = caption_layout_bbox
                # Figure out if the block satified for being the table caption of footnote condition
                if (block_center[0] > layout_bbox[0] and
                    block_center[0] < layout_bbox[2] and
                    block_center[1] > layout_bbox[1] and
                    block_center[1] < layout_bbox[3]):
                    selected_caption_block = block_objs.pop(i)
                    break
                i += 1
            
            if selected_caption_block:
                image_block.update_bbox(selected_caption_block.bbox)
                image_block.update_title(selected_caption_block.content)
                image_block.update_title_fonts([span.font for span in selected_caption_block.spans])
                
        block_objs.extend(image_blocks)
        return layout_infos, block_objs


    def merge_same_font_blocks(self, block_objs: List[Block]):
        refined_blocks: List[Block] = list()
        start_index = 0
        for i in range(0, len(block_objs)):
            refined_blocks.append(block_objs[i])
            if block_objs[i].type == BlockType.TEXT:
                start_index = i + 1
                break
        if len(block_objs) > 1:
            for i in range(start_index, len(block_objs)):
                block_obj = block_objs[i]
                if block_obj.type == BlockType.TEXT:
                    # find last text block and its index
                    last_block = None
                    index = -1
                    for i, l_block in enumerate(refined_blocks[::-1]):
                        if l_block.type == BlockType.TEXT:
                            last_block = l_block
                            index = len(refined_blocks) - i - 1
                            break
                    # merge block for different case of font and size
                    if (last_block.spans[-1].font == block_obj.spans[0].font and
                        last_block.block_no == block_obj.block_no):
                        if len([line for line in block_obj.lines if line not in last_block.lines]) > 0:
                            last_block.update_content(f"\n{block_obj.content.strip()}")
                        else :
                            last_block.update_content(block_obj.content)
                        last_block.update_content_fonts([span.font for span in block_obj.spans])
                        last_block.update_lines(block_obj.lines)
                        last_block.update_block_spans(block_obj.spans)
                        last_block.update_bbox_using_spans(block_obj.spans)
                    else:
                        refined_blocks.append(block_obj)
                    if index != -1:
                        refined_blocks[index] = last_block
                elif block_obj.type in {BlockType.IMAGE, BlockType.TABLE}:
                    refined_blocks.append(block_obj)
        return refined_blocks

    def merge_blocks_based_on_title(self, block_objs: List[Block]):
        text_blocks: List[Block] = []
        other_blocks: List[Block] = []
        for block in block_objs:
            if block.type == BlockType.TEXT:
                text_blocks.append(block)
            else:
                other_blocks.append(block)
        
        refined_text_blocks: List[Block] = []
        for text_block in text_blocks:
            if text_block.content != None :
                # get last block with title index
                lbwt_i = -1
                for i, l_block in enumerate(refined_text_blocks[::-1]):
                    if l_block.title != None :
                        lbwt_i = len(refined_text_blocks) - i - 1
                        break
                if lbwt_i != -1:
                    refined_text_blocks[lbwt_i].update_block(text_block)
                else :
                    refined_text_blocks.append(text_block)
            else:
                refined_text_blocks.append(text_block)
        
        refined_text_blocks.extend(other_blocks)
        return refined_text_blocks


    def merge_blocks_for_none_title(self, block_objs: List[Block]):
        i = 0
        first_block = None
        if block_objs[i].title == None:
            first_block = block_objs[i]
        i = 1
        while(i < len(block_objs) and
              block_objs[i].type == BlockType.TEXT and
              block_objs[i].title == None):
            first_block.update_block(block_objs.pop(i))
        return block_objs


    def merge_blocks_based_on_block(self, block_objs: List[Block]):
        refined_blocks: List[Block] = list()
        start_index = 0
        for i in range(0, len(block_objs)):
            refined_blocks.append(block_objs[i])
            if block_objs[i].type == BlockType.TEXT:
                start_index = i + 1
                break
        if len(block_objs) > 1:
            for i in range(start_index, len(block_objs)):
                block_obj = block_objs[i]
                if block_obj.type == BlockType.TEXT:
                    # find last text block and its index
                    last_block = None
                    index = -1
                    for i, l_block in enumerate(refined_blocks[::-1]):
                        if l_block.type == BlockType.TEXT:
                            last_block = l_block
                            index = len(refined_blocks) - i - 1
                            break
                    # here it check condition if type is in font and it doesn't contain 'bold' then
                    # merge block if font name and size are same and both block are from same pymupdf block
                    if (
                        (not hasattr(last_block.spans[-1].font, 'type') or
                         getattr(last_block.spans[-1].font.type, '', '').lower() != 'bold') and
                        (not hasattr(block_obj.spans[0].font, 'type') or
                         getattr(block_obj.spans[0].font.type, '', '').lower() != 'bold') and
                        last_block.block_no == block_obj.block_no
                    ):
                        if len([line for line in block_obj.lines if line not in last_block.lines]) > 0:
                            last_block.update_content(f"\n{block_obj.content}")
                        else :
                            last_block.update_content(block_obj.content)
                        last_block.update_lines(block_obj.lines)
                        last_block.update_content_fonts([span.font for span in block_obj.spans])
                        last_block.update_block_spans(block_obj.spans)
                        last_block.update_bbox_using_spans(block_obj.spans)
                    else :
                        refined_blocks.append(block_obj)
                    if index != -1:
                        refined_blocks[index] = last_block
                elif block_obj.type in {BlockType.IMAGE, BlockType.TABLE}:
                    refined_blocks.append(block_obj)
        return refined_blocks


    def remove_empty_blocks(self, block_objs: List[Block]):
        return [block for block in block_objs if 
                ((block.content and block.content.strip()) or
                (block.title and block.title.strip())) and
                block.type in {BlockType.TEXT}
                ] 

    def clean_block_objs(self, layout_infos: List[Dict], block_objs: List[Block]):
        block_objs = self.remove_empty_blocks(block_objs)
        block_objs = self.merge_same_font_blocks(block_objs)
        layout_infos, block_objs = self.remove_unnecessary_layout(
                                            layout_infos,
                                            block_objs,
                                            labels = [LayoutLabel.ABANDON]
                                            )
        layout_infos, block_objs = self.add_image_table_blocks(
                                            layout_infos,
                                            block_objs
                                        )
        
        block_objs = self.remove_table_blocks(block_objs)
        block_objs = self.remove_image_blocks(block_objs)
        block_objs = self.merge_blocks_based_on_block(block_objs)
        
        layout_infos, block_objs = self.update_table_caption(layout_infos, block_objs)
        layout_infos, block_objs = self.update_image_caption(layout_infos, block_objs)
        layout_infos, block_objs = self.update_formula_blocks(layout_infos, block_objs)
        layout_infos, block_objs = self.update_title_blocks(layout_infos, block_objs)
        
        block_objs = self.merge_blocks_based_on_title(block_objs)
        block_objs = self.merge_blocks_for_none_title(block_objs)
        # layout_infos, block_objs = self.update_image_table_caption(layout_infos, block_objs)
        
        return block_objs
    