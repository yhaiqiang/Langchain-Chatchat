# -*- coding: utf-8 -*-
# @Time    : 2023/8/14 14:17
# @Author  : qiang
# @File    : my_pdf_loader.py
# @Software: PyCharm
import copy
import json
import re

import pdfplumber
from langchain.document_loaders.parsers.pdf import PDFPlumberParser
from langchain.document_loaders import PDFPlumberLoader
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter,SpacyTextSplitter
import pandas as pd
from tabulate import tabulate


class MyPDFPlumberParser(PDFPlumberParser):

    def is_page_number(self, text_block, page_height):
        # 判断文本块是否位于页码区域
        threshold = 0.9 * page_height  # 调整此阈值以适应具体情况
        return text_block['top'] > threshold

    def extract_words_and_table(self, page, table_region):
        page_content = []
        page_height = page.height
        flag = True
        for text_block in page.extract_words():
            text_coordinates = text_block['x0'], text_block['top'], text_block['x1'], text_block['bottom']
            if not self.is_inside_table(table_region, text_coordinates) and not self.is_page_number(text_block, page_height):  # 自定义函数判断文本块是否在表格区域内
                page_content.append(text_block['text'])
                flag = True
            else:
                if flag and not self.is_page_number(text_block, page_height):
                    process_table = self.process_table(page)
                    # page_content.append("===表格开始==="+process_table+"===表格结束===")
                    page_content.append(process_table)
                    flag = False
        return '\n'.join(page_content)

    def is_inside_table(self, table_coordinates, text_coordinates):
        text_x0, text_top, text_x1, text_bottom = text_coordinates
        table_x0, table_y0, table_x1, table_y1 = table_coordinates

        if (text_x0 >= table_x0 and text_x1 <= table_x1 and
                text_top >= table_y0 and text_bottom <= table_y1):
            return True
        else:
            return False

    def find_table_coordinates(self, table_data):
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        for cell in table_data.edges:
            x, y, width, height = cell['x0'], cell['top'], cell['width'], cell['height']
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + width)
            max_y = max(max_y, y + height)

        return min_x, min_y, max_x, max_y

    def convert_table(self, df):
        num_list = []
        for column_name in df.columns:
            column_data = df[column_name]
            n = 0
            if len(column_data) >= 2:
                for i in range(len(column_data) - 1):
                    if column_data[i] != '' and column_data[i + 1] == column_data[i] and not re.match("\d+|\s+", column_data[i]):
                        n += 1
                        if n == len(column_data) - 1:
                            num_list.append(n + 1)
                    else:
                        num_list.append(n + 1)
                        break

        # 获取可能为表头的最大行数
        max_num = max(num_list)
        if max_num == 2:
            new_header = []
            for n in range(max_num - 1):
                new_header = [str(y)+str(x) if str(x)!=str(y) else str(x) for x, y in zip(df.iloc[n].tolist(), df.iloc[n + 1].tolist())]
            df.columns = new_header
            df = df[max_num:]
        elif max_num == 1:
            new_header = df.iloc[0]
            df.columns = new_header
            df = df[max_num:]

        # 识别表头并组合
        try:
            header = [re.sub('\s+|\n+', '', x) for x in df.columns]
        except:
            header = ["国家（地区)","城市", "币种", "住宿费(每人每天)", "伙食费(每人每天)", "公杂费(每人每天)"]
        result = []

        # 结构化数据归为非结构化数据
        for index, row in df.iterrows():
            row = [re.sub('\s+|\n+', '', x) for x in row]
            tmp=[]
            for i in range(len(header)):
                if row[i]=="-":
                    row[i]="无"
                tmp.append(header[i]+row[i])
            result.append(tmp)
        # result_merge = "。".join(result)
        result_str = str(result).replace("'", "").replace(":", "")
        return result_str

    def process_table(self, page):
        tables = page.extract_tables()
        result_list = []
        for table in tables:
            df = pd.DataFrame(table)
            for column_name in df.columns:
                column_data = df[column_name]
                if len(column_data) >= 2:
                    for i in range(len(column_data) - 1):
                        if column_data[i] is not None and column_data[i + 1] is None:
                            column_data[i + 1] = column_data[i]
            for index, row in df.iterrows():
                if len(row) >= 2:
                    for i in range(len(row) - 1):
                        if row[i] is not None and row[i + 1] is None:
                            row[i + 1] = row[i]
            # result = self.convert_table(df)
            result = self.convert_table_to_markdown(df)
            result_list.append(result)
        return "\n\n".join(result_list)

    def convert_table_to_markdown(self,df):
        markdown_table = tabulate(df, headers='keys', tablefmt='pipe',showindex=False)
        print(tabulate)
        return markdown_table

    def extract_content(self, page):
        table = page.debug_tablefinder()
        table_coordinates = self.find_table_coordinates(table)
        table_x1, table_y1 = table_coordinates[0], table_coordinates[1]
        table_x2, table_y2 = table_coordinates[2], table_coordinates[3]
        table_region = (table_x1, table_y1, table_x2, table_y2)
        # 表格区域坐标，需要自行设定
        content = self.extract_words_and_table(page, table_region)
        return content


    def lazy_parse(self, blob: Blob):
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:
            doc = pdfplumber.open(file_path)  # open document

            page_doc = []

            for i, page in enumerate(doc.pages):
                page_content = self.extract_content(page)
                print(page_content)
                metadata = dict({"source": blob.source, "file_path": blob.source, "page": page.page_number,"total_pages": len(doc.pages)},**{k: doc.metadata[k] for k in doc.metadata if type(doc.metadata[k]) in [str, int]})
                doc_object=Document(page_content=page_content, metadata=metadata)
                page_doc.append(doc_object)
                print(f"--------------分页线{i+1}-------------")
            yield from page_doc

            # yield from [
            #     Document(
            #         page_content=self.extract_content(page),
            #         metadata=dict(
            #             {
            #                 "source": blob.source,
            #                 "file_path": blob.source,
            #                 "page": page.page_number,
            #                 "total_pages": len(doc.pages),
            #             },
            #             **{
            #                 k: doc.metadata[k]
            #                 for k in doc.metadata
            #                 if type(doc.metadata[k]) in [str, int]
            #             },
            #         ),
            #     )
            #     for page in doc.pages[:1]
            # ]

class MyPDFPlumberParser2(PDFPlumberParser):
    def extract_content(self, page):
        tables = page.extract_tables(table_settings={"vertical_strategy": "lines", "snap_tolerance": 10,"horizontal_strategy": "lines"})[0]
        tables=[list(filter(lambda x: x is not None and x != "",each)) for each in tables]
        tables_filter=[]
        name = tables[1][1]
        # name = tables[0][0].replace(" ","").replace("产品规格书","")
        for each in tables[2:]:
            if len(each)==1:
                continue
            else:
                # if "接口明细" in each:
                #     for x in each[2].split("、"):
                #         text=name+"".join(each[:2])+x+"支持"
                #         text = text.replace("\n", "")
                #         tables_filter.append(text)
                # else:
                    text ="".join(each)
                    # if each[-1]=="无":
                    #     text+="、不支持"
                    # if each[-1]=="是":
                    #     text+="、支持"
                    if "宽厚高" in text or "壁挂孔距" in text:
                        text = text.replace("尺寸与重量", "")
                        text = "尺寸"+text
                    text = name+text
                    text = text.replace("\n", "")
                    tables_filter.append(text)
        return "\n\n".join(tables_filter)

    def lazy_parse(self, blob: Blob):
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:
            doc = pdfplumber.open(file_path)  # open document

            page_doc = []

            for i, page in enumerate(doc.pages):
                page_content = self.extract_content(page)
                print(page_content)
                metadata = dict({"source": blob.source, "file_path": blob.source, "page": page.page_number,"total_pages": len(doc.pages)},**{k: doc.metadata[k] for k in doc.metadata if type(doc.metadata[k]) in [str, int]})
                doc_object=Document(page_content=page_content, metadata=metadata)
                page_doc.append(doc_object)
                print(f"--------------分页线{i+1}-------------")
            yield from page_doc


class MyPDFPlumberLoader(PDFPlumberLoader):
        """ """
        def load(self):
                parser =MyPDFPlumberParser2(text_kwargs=self.text_kwargs)
                blob = Blob.from_path(self.file_path)
                return parser.parse(blob)



if __name__ == '__main__':
    parse = MyPDFPlumberLoader(file_path=r"虚拟导购产品信息库.pdf")
    res=parse.load()
    print(res)
