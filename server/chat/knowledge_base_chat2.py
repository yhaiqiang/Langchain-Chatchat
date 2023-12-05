import csv
from pprint import pprint
from jinja2 import Template
from fastapi import Body, Request
import re
import  time
from fastapi.responses import StreamingResponse
from configs import (LLM_MODELS, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE)
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.utils import get_doc_path
import json
from pathlib import Path
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from sse_starlette.sse import EventSourceResponse

# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks

# pipeline_ins = pipeline(task=Tasks.text2text_generation, model='damo/nlp_mt5_dialogue-rewriting_chinese-base', model_revision='v1.0.1')
tmp_dict = {}


async def knowledge_base_chat2(query: str = Body(..., description="用户输入", examples=["你好"]),
                            knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                            top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                            score_threshold: float = Body(SCORE_THRESHOLD, description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右", ge=0, le=2),
                            history: List[History] = Body([],
                                                      description="历史对话",
                                                      examples=[[
                                                          {"role": "user",
                                                           "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                          {"role": "assistant",
                                                           "content": "虎头虎脑"}]]
                                                      ),
                            stream: bool = Body(False, description="流式输出"),
                            model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                            temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                            max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                            prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                            request: Request = None,
                        ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    with open("/home/haiqiang/Projects/Langchain-Chatchat/knowledge_base/虚拟导购知识库/knowl.json") as f:
        line_dict={}
        for x in f:
            line_dict.update(json.loads(x))

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(query: str,
                                           top_k: int,
                                           history: Optional[List[History]],
                                           model_name: str = LLM_MODELS[0],
                                           prompt_name: str = prompt_name,
                                           ) -> AsyncIterable[str]:

        # query改写
        # product_name=""
        # product_name_asistant=""
        # for h in history:
        #     if h.role=="assistant":
        #     #     matched=re.search("(43f3|43g6a|(55|65|75)?(r6|x3)|98c2|led86k2|65r6|55g5u)", h.content.lower())
        #     #     # matched=re.search("([a-zA-Z0-9]+)", h.content.lower())
        #     #     if matched:
        #     #         product_name_asistant=matched.group(0)
        #         h.content=re.sub("{.*?}","",h.content)
        #     if h.role=="user":
        #         matched=re.search("(43f3|43g6a|(55|65|75)?(r6|x3)|98c2|led86k2|65r6|55g5u)", h.content.lower())
        #         # matched=re.search("([a-zA-Z0-9]+)", h.content.lower())
        #         if matched:
        #             product_name=matched.group(0)
        # if not product_name:
        #     product_name=product_name_asistant
        # if product_name:
        #     query=f"产品型号{product_name}"+query
        #     # query="product_name"+query

        # 首先对原问题进行检索
        t0=time.time()
        docs = search_docs(query, knowledge_base_name, top_k, score_threshold) #返回的是DocumentWithScore对象
        print("向量匹配时间：",time.time()-t0)
        pprint(docs)

        # 如果没有历史会话且原问题未检索到答案则将问题改写后再进行检索
        if history and not docs:
            t0 = time.time()
            callback = AsyncIteratorCallbackHandler()
            model = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=[callback],
            )
            prompt_template = get_prompt_template("knowledge_base_chat", "query_rewrite")
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                # [i.to_msg_template() for i in history] +
                [input_msg])
            chain = LLMChain(prompt=chat_prompt, llm=model)

            context = ""
            # context = []
            for h in history:
                # context.append({"role":h.role,"content":h.content})
                if h.role == "user":
                    context +=  h.content + "[SEP]"
                if h.role == "assistant":
                    context += h.content + "[SEP]"
            # context+=query
            context=json.dumps(context,ensure_ascii=False)
            pprint("原始上下文：" + str(context))
            for k, v in tmp_dict.items():
                context = context.replace(k, v)
            pprint("替换后的上下文：" + context)

            _task = asyncio.create_task(wrap_done(
                chain.acall({"context": context,"question":query}),
                callback.done),
            )
            _answer = ""
            async for token in callback.aiter():
                _answer += token
            await _task
            pprint("改写后的结果:"+ _answer)
            _answer=_answer.replace("输出:","").replace("输出：","")
            tmp_dict[query] = _answer
            query = _answer
            print("改写消耗时间：",time.time()-t0)

            docs = search_docs(query, knowledge_base_name, top_k, score_threshold) #返回的是DocumentWithScore对象



        # 将召回的问题替换为对应的答案:
        answer_id=[]
        docs_new=[]
        for doc in docs:
            if doc.page_content in line_dict and line_dict[doc.page_content]["知识库ID"] not in answer_id:
                answer_id.append(line_dict[doc.page_content]["知识库ID"])
                doc.page_content=line_dict[doc.page_content]["回答1"]+";"+line_dict[doc.page_content]["回答2"]
                docs_new.append(doc)
        docs=docs_new if docs_new else docs
        pprint(docs)

        # 过滤出置信度比较高的召回的文档
        docs_new=[]
        docs=sorted(docs,key=lambda x:x.score,reverse=False)
        if docs and docs[0].score<=0.2:
            docs_new.append(docs[0])
        docs=docs_new if docs_new else docs

        # csv文件解析成字典格式的的文本提取出对应的字段
        docs_filter=[]
        for doc in docs:
            text=doc.page_content
            element_list = text.split("\n")
            if "image_url" in text:
                for x in element_list:
                    if "image_url" in x:
                        url=x.split(":")[-1]
                        if url.strip()!="":
                            yield json.dumps({"answer": json.dumps({"image_url:":url},ensure_ascii=False)})+"\n\n"
            if "回答" in text:
                ans_list=[]
                for x in element_list:
                    if "回答" in x:
                        ans_text=x.split(":",1)[-1]
                        if ans_text:
                            ans_list.append(ans_text)
                doc.page_content="\n".join(ans_list)

        # docs = search_by_knowl_dict(query, knowl_dict)
        #
        # if not docs:
        #     yield json.dumps({"answer": "知识库暂无此信息，无法回答该问题"}, ensure_ascii=False)
        #     return  # 停止后续处理
        #
        # if "支不支持" in query:
        #     query=query.replace("支不支持","支持还是不支持")

        # docs_filter = []
        # for doc in docs:
        #     product_name = re.search("([0-9a-zA-Z]+)", query)
        #     if product_name:
        #         product_name = product_name.group(0)
        #         if product_name.lower() in doc.page_content.lower():
        #             docs_filter.append(doc)
        # docs = docs_filter

        def mt5_rewrite(query,context):
            # mt5模型改写
            query = query.strip().strip("。")
            history_content = ''
            for i, h in enumerate(history):
                h.content = h.content.strip().strip("。")
                joint_symbol = "[SEP]" if i < len(history) - 1 else ""
                history_content += h.content + joint_symbol
            print("历史对话：", history_content)

            for k, v in tmp_dict.items():
                history_content = history_content.replace(k, v)
            print("历史对话替换：", history_content)

            history_and_current_content = history_content + "[SEP]" + query
            print("对话改写的输入：", history_and_current_content)

            query_rewrite = pipeline_ins(input=history_and_current_content).get("text", "")
            if query_rewrite != query:
                tmp_dict[query] = query_rewrite
            print("对话改写的输出：", query)


            # # 利用大模型判断是否需要进行指代消解
            # callback = AsyncIteratorCallbackHandler()
            # model = get_ChatOpenAI(
            #     model_name=model_name,
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     callbacks=[callback],
            # )
            # prompt_template = get_prompt_template("knowledge_base_chat", "is_coreference_resolution")
            # # template = Template(prompt_template)
            # # result = template.render(history_content=history_content, question=question)
            # input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            # chat_prompt = ChatPromptTemplate.from_messages(
            #     # [i.to_msg_template() for i in history] +
            #     [input_msg])
            # chain = LLMChain(prompt=chat_prompt, llm=model)
            # pprint("指代消解的prompt")
            # pprint(chain)
            # # Begin a task that runs in the background.
            # _task = asyncio.create_task(wrap_done(
            #     chain.acall({"context": context, "question": query}),
            #     callback.done),
            # )
            # _answer = ""
            # async for token in callback.aiter():
            #     _answer += token
            # await _task
            # print("是否指代消解的结果", _answer)
            #
            #
            # # 利用大模型结合历史对话改写当下的问题
            # # if _answer=="1":
            # callback = AsyncIteratorCallbackHandler()
            # model = get_ChatOpenAI(
            #     model_name=model_name,
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     callbacks=[callback],
            # )
            # prompt_template = get_prompt_template("knowledge_base_chat", "query_rewrite")
            # # template = Template(prompt_template)
            # # result = template.render(history_content=history_content, question=question)
            # input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            # chat_prompt = ChatPromptTemplate.from_messages(
            #     # [i.to_msg_template() for i in history]+
            #     [input_msg])
            # chain = LLMChain(prompt=chat_prompt, llm=model)
            # pprint("query改写的prompt")
            # pprint(chain)
            # # Begin a task that runs in the background.
            # _task = asyncio.create_task(wrap_done(
            #     chain.acall({"context": context, "question": query}),
            #     callback.done),
            # )
            # _answer = ""
            # async for token in callback.aiter():
            #     _answer += token
            # await _task
            # query=_answer
            # print("query改写", _answer)

        context = "\n".join([doc.page_content for doc in docs])

        # 先用大模型做一个意图识别
        # callback = AsyncIteratorCallbackHandler()
        # model = get_ChatOpenAI(
        #     model_name=model_name,
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        #     callbacks=[callback],
        # )
        # prompt_template = get_prompt_template("knowledge_base_chat", "intent_classifier")
        # input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        # chat_prompt = ChatPromptTemplate.from_messages(
        #     # [i.to_msg_template() for i in history]+
        #     [input_msg])
        # chain = LLMChain(prompt=chat_prompt, llm=model)
        # pprint("意图识别的prompt")
        # pprint(chain)
        # # Begin a task that runs in the background.
        # _task = asyncio.create_task(wrap_done(
        #     chain.acall({"context": context, "question": query}),
        #     callback.done),
        # )
        # _answer = ""
        # async for token in callback.aiter():
        #     _answer += token
        # await _task
        # print("intent",_answer)

        if len(docs) == 0: ## 如果没有找到相关文档，使用Empty模板
            # if _answer == "1":
                yield json.dumps({"answer": "知识库暂无此信息，无法回答该问题"}, ensure_ascii=False)
                return  # 停止后续处理
            # else:
            # prompt_template = get_prompt_template("knowledge_base_chat", "Empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        pprint("知识库问答的prompt")
        pprint(chat_prompt)

        callback = AsyncIteratorCallbackHandler()#这个迭代器被上面的任务用完了，所以要重新再初始化一次
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        doc_path = get_doc_path(knowledge_base_name)
        for inum, doc in enumerate(docs):
            filename = Path(doc.metadata["source"]).resolve().relative_to(doc_path)
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name":filename})
            base_url = request.base_url
            base_url=str(base_url).replace("127.0.0.1","172.19.11.1")
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0: # 没有找到相关文档
            source_documents.append(f"""<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>""")

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task


    # return StreamingResponse(knowledge_base_chat_iterator(query=query,
    return EventSourceResponse(knowledge_base_chat_iterator(query=query,
                                                          top_k=top_k,
                                                          history=history,
                                                          model_name=model_name,
                                                          prompt_name=prompt_name),
                             media_type="text/event-stream")


def load_knowl_dict(file_path):
    with open(file_path, encoding="utf-8") as json_file:
        knowl_dict = json.load(json_file)
    return knowl_dict