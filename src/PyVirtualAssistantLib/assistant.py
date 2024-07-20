import multiprocessing
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatLlamaCpp
from langchain.memory import ConversationSummaryMemory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from duckduckgo_search import DDGS

from .screen_interaction import ScreenInteractor

accepted_roles = ["human", "system", "assistant"]


class Model:
    def __init__(
            self,
            model_path: str,
            language: str = "en",
            system_prompt: str = "You are a helpful assistant. Answer any questions to the best of your ability.",
            temperature: float = 0.5,
            n_ctx: int = 10000,
            n_gpu_layers: int = 8,
            n_batch: int = 300,
            max_tokens: int = 512,
            n_threads: int = multiprocessing.cpu_count() - 1,
            repeat_penalty: float = 1.5,
            top_p: float = 0.5,
            verbose: bool = False
    ):
        self.model_path = model_path
        self.__llm = ChatLlamaCpp(
            temperature=temperature,
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            max_tokens=max_tokens,
            n_threads=n_threads,
            repeat_penalty=repeat_penalty,
            top_p=top_p,
            verbose=False#verbose
        )

        self.verbose = verbose
        self.__messages = [("system", system_prompt)]
        self.__ddg_search = DDGS()  # search engine for if no relevant docs are found
        self.__screen_interactor = ScreenInteractor(language)

        # init embeddings
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = Chroma(embedding_function=self.embeddings)

        # Initialize retriever
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        self.retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainExtractor.from_llm(self.__llm),
            base_retriever=base_retriever
        )

        self.__memory = ConversationSummaryMemory(
            llm=self.__llm, memory_key="chat_history", return_messages=True
        )

    def add_documents(self, documents: List[Document]):
        for doc in documents:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = 'Unknown'
        self.vector_store.add_documents(documents)

    def add_web_documents(self, query: str, num_results: int = 3):
        search_results = self.__ddg_search.text(query, max_results=num_results)
        print("SEARCH RESULTS: ", search_results)
        self.add_documents([
            Document(
                page_content=f"Title: {result['title']}\n\nBody: {result['body']}",
                metadata={"source": result['href']}
            ) for result in search_results
        ])

    def chat(self, text: str, role: str = "human") -> str:
        if role not in accepted_roles:
            raise ValueError(f"Role must be one of {accepted_roles}")

        self.__messages.append((role, text))

        all_docs = self.retriever.invoke(text)

        print("ALLDOCS1: ", all_docs)
        relevant_docs = [doc for doc in all_docs if not ("NO_OUTPUT" in doc.page_content)]

        if not relevant_docs:
            print("Searching the web for relevant information...")
            self.add_web_documents(text, num_results=2)
            all_docs = self.retriever.invoke(text)
            print("ALLDOCS2: ", all_docs)
            relevant_docs = [doc for doc in all_docs if not ("NO_OUTPUT" in doc.page_content)]

        if relevant_docs:
            context = "\n".join([doc.page_content for doc in relevant_docs])
            self.__messages.append(("system", f"Relevant context:\n{context}"))

        msg: BaseMessage = self.__llm.invoke(self.__messages)
        content: str = msg.content

        self.__messages.append(("assistant", content))

        print("AI: " + content)

        if self.verbose:
            print(f"MESSAGES: {self.__messages}")

        return content
