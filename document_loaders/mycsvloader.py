# -*- coding: utf-8 -*-
# @Time    : 2023/10/30 14:07
# @Author  : qiang
# @File    : mycsvloader.py
# @Software: PyCharm
import json
import os
from io import TextIOWrapper
import csv

from langchain.document_loaders.csv_loader import CSVLoader
from typing import Any, Dict, List, Optional, Sequence
from langchain.docstore.document import Document
from langchain.document_loaders.helpers import detect_file_encodings


class MyCSVLoader(CSVLoader):
    def load(self) -> List[Document]:
        """Load data into document objects."""
        docs = []
        try:
            with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
                docs = self.__read_file(csvfile)
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    try:
                        with open(
                                self.file_path, newline="", encoding=encoding.encoding
                        ) as csvfile:
                            docs = self.__read_file(csvfile)
                            break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        return docs

    def __read_file(self, csvfile: TextIOWrapper) -> List[Document]:
        docs = []

        question_answer_dict = {}
        csv_reader = csv.DictReader(csvfile, **self.csv_args)  # type: ignore
        column_name=csv_reader.fieldnames
        for i, row in enumerate(csv_reader):
            try:
                source = (
                    row[self.source_column]
                    if self.source_column is not None
                    else self.file_path
                )
            except KeyError:
                raise ValueError(
                    f"Source column '{self.source_column}' not found in CSV file."
                )
            content = "\n".join(
                f"{k.strip()}: {v.strip()}"
                for k, v in row.items()
                if k not in self.metadata_columns
            )
            metadata = {"source": source, "row": i}
            for col in self.metadata_columns:
                try:
                    metadata[col] = row[col]
                except KeyError:
                    raise ValueError(f"Metadata column '{col}' not found in CSV file.")
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)

            # question_list=row["相似问"].split("||")
            # question_list.insert(0,row["标准问"])
            # answer_list=[]
            # tmp_dict={}
            # for x in column_name:
            #     if x not in ["知识库ID","相似问","标准问"] and row[x] !="":
            #         answer_list.append(row[x])
            #
            # tmp_dict["answer"]=answer_list
            # tmp_dict["knowl_id"]=str(row["知识库ID"])
            #
            # for x in question_list:
            #     question_answer_dict[x]=tmp_dict

        # current_dir = os.path.dirname(os.path.dirname(os.path.abspath(self.file_path)))
        # with open(os.path.join(current_dir,"knowl.json"),"w",encoding="utf-8") as json_file:
        #     json.dump(question_answer_dict,json_file,ensure_ascii=False,indent=4)

        return docs

if __name__ == '__main__':
    parse = MyCSVLoader(file_path=r"/root/Projects/Langchain-Chatchat/knowledge_base/虚拟导购知识库/content/虚拟导购_部分问答对.csv",encoding='utf-8')
    res=parse.load()
    for x in res:
        print(x)
