import multiprocessing
from typing import List, Iterator

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatLlamaCpp
from langchain.memory import ConversationSummaryMemory
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.documents import Document

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
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        embeddings_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.7)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=base_retriever
        )

        self.__memory = ConversationSummaryMemory(llm=self.__llm, memory_key="chat_history", return_messages=True)

    def add_documents(self, documents: List[Document]):
        for doc in documents:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = 'Unknown'
        self.vector_store.add_documents(documents)

    def add_web_documents(self, query: str, num_results: int = 3):
        search_results = self.__ddg_search.text(query, max_results=num_results)
        if self.verbose:
            print("SEARCH RESULTS: ", search_results)
        self.add_documents([
            Document(
                page_content=f"Title: {result['title']}\n\nBody: {result['body']}",
                metadata={"source": result['href']}
            ) for result in search_results
        ])

    def chat(self, text: str, role: str = "human") -> Iterator[str]:
        if role not in accepted_roles:
            raise ValueError(f"Role must be one of {accepted_roles}")

        # Retrieve relevant documents
        all_docs = self.retriever.invoke(text)

        if not all_docs:
            if self.verbose:
                print("No relevant documents found. Searching the web...")
            self.add_web_documents(text, num_results=1)
            all_docs = self.retriever.invoke(text)

        # Prepare context
        if all_docs:
            context = "\n".join([doc.page_content for doc in all_docs])
        else:
            context = "No relevant information found."

        # Add conversation history to messages
        chat_history = self.__memory.load_memory_variables({})["chat_history"]

        # Prepare messages for the LLM
        messages = [
            ("system", self.__messages[0][1]),  # System prompt
            ("system", f"Relevant context:\n{context}"),
            *chat_history,
            (role, text)
        ]

        # Generate response
        stream = self.__llm.stream(messages)

        full_response = ""
        for chunk in stream:
            content = chunk.content
            full_response += content
            yield content

        # Save the conversation to memory
        self.__memory.save_context({"input": text}, {"output": full_response})

        if self.verbose:
            print(f"MESSAGES: {messages}")
            print(f"MEMORY: {self.__memory.load_memory_variables({})}")
