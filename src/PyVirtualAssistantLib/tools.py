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
