import multiprocessing
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from .tools import SourceSelectionTool, AssistantTool

accepted_roles = ["human", "system", "assistant"]

class Model:
    # Private variables
    __llm: ChatLlamaCpp
    __model_path: str
    __messages: list
    __tools: list = []
    __tool_functions: list = []
    __system_prompt: str
    __verbose: bool

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

        # init embeddings
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = Chroma(embedding_function=self.embeddings)

        # Initialize retriever
        base_retriever = self.vector_store.as_retriever()
        self.retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainExtractor.from_llm(self.__llm),
            base_retriever=base_retriever
        )

    def add_documents(self, documents: List[Document]):
        self.vector_store.add_documents(documents)

    def chat(self, text: str, role: str = "human") -> str:
        if not (role in accepted_roles):
            raise ValueError(f"Role must be one of {accepted_roles}")

        self.__messages.append((role, text))

        source_selection_tool = next((tool for tool in self.__tools if isinstance(tool, SourceSelectionTool)), None)
        if source_selection_tool:
            available_sources = [doc.metadata['source'] for doc in self.vector_store.get()]
            selected_sources = source_selection_tool.tool_function(text, available_sources)

            relevant_docs = self.retriever.get_relevant_documents(text)
            filtered_docs = [doc for doc in relevant_docs if doc.metadata['source'] in selected_sources]
        else:
            filtered_docs = self.retriever.get_relevant_documents(text)

        context = "\n".join([doc.page_content for doc in filtered_docs])
        self.__messages.append(("system", f"Relevant context:\n{context}"))

        msg: BaseMessage = self.__llm.invoke(self.__messages)
        content: str = msg.content

        # Use tools to enhance the response if applicable
        for tool in self.__tools:
            if tool.get_tool_name() in content:
                try:
                    tool_result = tool.tool_function(text)
                    content += f"\n\nAdditional information from {tool.get_tool_name()}:\n{tool_result}"
                except Exception as e:
                    content += f"\n\nError using {tool.get_tool_name()}: {str(e)}"

        self.__messages.append(("assistant", content))

        print("AI: " + content)

        if self.__verbose:
            print(self.__messages)

        return content

    def bind_tool(self, tool_class: AssistantTool):
        self.__tools.append(tool_class)
        self.__llm = self.__llm.bind_tools(
            tools=[tool_class.tool_function],
            tool_choice={"type": "function", "function": {"name": tool_class.get_tool_name()}}
        )

        if self.__verbose:
            print(tool_class.get_tool_name() + " bound to model")

    def bind_tools(self, tool_classes: list):
        for tool_class in tool_classes:
            self.bind_tool(tool_class)