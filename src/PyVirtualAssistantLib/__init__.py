from .assistant import Model
from langchain_core.documents import Document
from typing import List

__all__ = ['create_assistant', 'add_documents', 'add_web_documents', 'Model', 'Document']


def create_assistant(model_path, verbose=False):
    model = Model(model_path, verbose=verbose)
    return model


def add_documents(assistant: Model, documents: List[Document]):
    assistant.add_documents(documents)

def add_web_documents(assistant: Model, query: str, num_results: int = 3):
    assistant.add_web_documents(query, num_results)