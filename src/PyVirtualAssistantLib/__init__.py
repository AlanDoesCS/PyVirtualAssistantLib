from .assistant import Model
from .tools import WeatherTool, WebScraperTool, WebSearchTool, SourceSelectionTool
from langchain_core.documents import Document
from typing import List

__all__ = ['create_assistant', 'Model', 'WeatherTool', 'WebScraperTool', 'WebSearchTool', 'SourceSelectionTool', 'Document']

def create_assistant(model_path, verbose=False):
    model = Model(model_path, verbose=verbose)
    model.bind_tool(WeatherTool())
    model.bind_tool(SourceSelectionTool())
    return model

def add_documents(assistant: Model, documents: List[Document]):
    assistant.add_documents(documents)