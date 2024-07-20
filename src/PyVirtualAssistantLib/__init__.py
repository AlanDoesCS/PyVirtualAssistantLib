from .assistant import Model
from langchain_core.documents import Document
from typing import List

__all__ = ['create_assistant', 'add_documents', 'Model', 'Document']


def create_assistant(model_path, verbose=False):
    model = Model(model_path, verbose=verbose)
    return model


def add_documents(assistant: Model, documents: List[Document]):
    assistant.add_documents(documents)
