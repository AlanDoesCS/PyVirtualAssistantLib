import multiprocessing
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from duckduckgo_search import DDGS

accepted_roles = ["human", "system", "assistant"]


class Model:
    def __init__(
            self,
            model_path: str,
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
            verbose=verbose
        )

        self.__verbose = verbose
        self.__messages = [("system", system_prompt)]
        self.__ddg_search = DDGS()  # search engine for if no relevant docs are found

        # init embeddings
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = Chroma(embedding_function=self.embeddings)

        # Initialize retriever
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        self.retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainExtractor.from_llm(self.__llm),
            base_retriever=base_retriever
        )

    def add_documents(self, documents: List[Document]):
        for doc in documents:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = 'Unknown'
        self.vector_store.add_documents(documents)

    def chat(self, text: str, role: str = "human") -> str:
        if role not in accepted_roles:
            raise ValueError(f"Role must be one of {accepted_roles}")

        self.__messages.append((role, text))

        all_docs = self.retriever.invoke(text)

        relevant_docs = [doc for doc in all_docs if not ("NO_OUTPUT" in doc.page_content)]

        if relevant_docs:
            context = "\n".join([doc.page_content for doc in relevant_docs])
            context_message = f"Relevant context:\n{context}"
        else:
            print("Searching the web for relevant information...")
            web_search_results = self.__ddg_search.text(text, max_results=2)
            context_message = f"No relevant information in provided documents.\n\n{web_search_results}"

        self.__messages.append(("system", context_message))

        msg: BaseMessage = self.__llm.invoke(self.__messages)
        content: str = msg.content

        self.__messages.append(("assistant", content))

        print("AI: " + content)

        if self.__verbose:
            print(self.__messages)

        return content
