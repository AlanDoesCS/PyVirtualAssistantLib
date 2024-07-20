from typing import List

from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field


# Test
class AssistantTool:
    tool_class: BaseModel
    tool_function: callable
    __name: str = None

    def __init__(self, name: str, tool_class: BaseModel, tool_function: callable):
        self.__name = name
        self.tool_class = tool_class
        self.tool_function = tool_function

    @classmethod
    def get_tool_name(cls) -> str:
        raise NotImplementedError("Subclasses should implement this method")


class WeatherTool(AssistantTool):
    class WeatherRequest(BaseModel):
        city: str = Field(..., title="City", description="The city to get the weather for")
        unit: str = Field(enum=["celsius", "fahrenheit"], default="celsius", title="Unit",
                          description="The unit to get the temperature in")

    @classmethod
    def get_tool_name(cls) -> str:
        return "get_current_weather"

    @tool("get_current_weather", args_schema=WeatherRequest)
    def get_current_weather(city: str, unit: str = "celsius") -> str:
        """
        Get the current weather for a city

        Args:
            city (str): The city to get the weather for.
            unit (str): The unit to get the temperature in (Celsius or Fahrenheit).

        Returns:
            str: The current temperature in the specified city and unit.
        """
        return f"The current temperature in {city} is 22 degrees {unit}"  # TODO: implement requests to weather api

    def __init__(self):
        super(WeatherTool, self).__init__(self.get_tool_name(), self.WeatherRequest, self.get_current_weather)

class WebScraperTool(AssistantTool):
    class WebsiteRequest(BaseModel):
        url: str = Field(..., title="URL", description="The specific webpage to scrape data from")

    @classmethod
    def get_tool_name(cls) -> str:
        return "get_website_data"

    @tool("get_website_data", args_schema=WebsiteRequest)
    def get_website_content(url: str) -> str:
        """
        Read the data from a webpage

        Args:
            URL (str): The webpage to scrape.

        Returns:
            str: The contents of that page
        """
        data = "Wikipedia page"
        return f"{url}:\n\n{data}"      # TODO: implement some form of scraping

    def __init__(self):
        super(WebScraperTool, self).__init__(self.get_tool_name(), self.WebsiteRequest, self.get_website_content)


class WebSearchTool(AssistantTool):
    class SearchEngineRequest(BaseModel):
        search_prompt: str = Field(..., title="SearchPrompt", description="The search prompt for the search engine")

    @classmethod
    def get_tool_name(cls) -> str:
        return "get_search_result"

    @tool("get_search_result", args_schema=SearchEngineRequest)
    def get_search_result(prompt: str) -> str:
        """
        Read the data from a webpage

        Args:
            prompt (str): The search prompt for the search engine.

        Returns:
            str: The contents of the best fitting search result
        """
        data = "Wikipedia page\nLorem ipsum dolor sit amet, consectetur adipiscing elit"
        return data      # TODO: implement some form of scraping

    def __init__(self):
        super(WebSearchTool, self).__init__(self.get_tool_name(), self.SearchEngineRequest, self.get_search_result)


class SourceSelectionTool(AssistantTool):
    class SourceSelectionRequest(BaseModel):
        query: str = Field(..., title="Query", description="The user's query")
        available_sources: List[str] = Field(..., title="Available Sources", description="List of available sources")

    @classmethod
    def get_tool_name(cls) -> str:
        return "select_sources"

    @tool("select_sources", args_schema=SourceSelectionRequest)
    def select_sources(query: str, available_sources: List[str]) -> List[str]:
        """
        Selects the relevant sources for a given query.

        Args:
            query (str): The user's query.
            available_sources (List[str]): List of available sources.

        Returns:
            List[str]: List of selected sources.
        """
        # TODO implement source selection logic
        return available_sources

    def __init__(self):
        super(SourceSelectionTool, self).__init__(self.get_tool_name(), self.SourceSelectionRequest, self.select_sources)