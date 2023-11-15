# -*- coding: utf-8 -*-
# @Time    : 2023/10/30 13:47
# @Author  : qiang
# @File    : bm25.py
# @Software: PyCharm
import math


# 定义BM25算法的参数
k1 = 1.5
b = 0.75

# # 示例文档集合
# documents = [
#     "This is the first document",
#     "This document is the second document",
#     "And this is the third one",
#     "Is this the first document?",
# ]
#
# # 查询字符串
# query = "first document"

# 计算文档中每个词的文档频率
def calculate_doc_frequencies(documents):
    doc_frequencies = {}
    for doc in documents:
        words = doc.split()
        for word in words:
            if word in doc_frequencies:
                doc_frequencies[word] += 1
            else:
                doc_frequencies[word] = 1
    return doc_frequencies

# 计算文档长度
def calculate_doc_length(doc):
    return len(doc.split())

# 计算文档平均长度
def calculate_avg_doc_length(documents):
    total_length = 0
    for doc in documents:
        total_length += calculate_doc_length(doc)
    return total_length / len(documents)

# 计算BM25分数
def calculate_bm25_score(query, doc, doc_frequencies, avg_doc_length,documents):
    words = query.split()
    score = 0
    doc_length = calculate_doc_length(doc)
    for word in words:
        if word in doc_frequencies:
            idf = math.log((len(documents) - doc_frequencies[word] + 0.5) / (doc_frequencies[word] + 0.5) + 1.0)
            numerator = doc_frequencies[word] * (k1 + 1)
            denominator = doc_frequencies[word] + k1 * (1 - b + b * (doc_length / avg_doc_length))
            score += idf * (numerator / denominator)
    return score

# 计算并排序文档
def retrieve_documents(query, documents):
    doc_frequencies = calculate_doc_frequencies(documents)
    avg_doc_length = calculate_avg_doc_length(documents)
    doc_scores = []

    for idx, doc in enumerate(documents):
        score = calculate_bm25_score(query, doc, doc_frequencies, avg_doc_length,documents)
        doc_scores.append((idx, score))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores

if __name__ == '__main__':
    pass
    # 运行检索
    # results = retrieve_documents(query, documents)

    # 打印检索结果
    # for idx, score in results:
    #     print(f"Document {idx + 1}: BM25 Score = {score:.4f}")
