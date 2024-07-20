from .assistant import Model
from .tools import WeatherTool, WebScraperTool, WebSearchTool, SourceSelectionTool

__all__ = ['Model', 'WeatherTool', 'WebScraperTool', 'WebSearchTool', 'SourceSelectionTool']

def create_assistant(model_path, verbose=False):
    model = Model(model_path, verbose=verbose)
    model.bind_tool(WeatherTool())
    model.bind_tool(SourceSelectionTool())
    return model