from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

# Test
class AssistantTool:
    tool_class = None

class WeatherRequest(BaseModel):
    city: str = Field(..., title="City", description="The city to get the weather for")
    unit: str = Field(enum=["celsius", "fahrenheit"], default="celsius", title="Unit",
                      description="The unit to get the temperature in")


@tool("get_current_weather", args_schema=WeatherRequest)
def get_current_weather(city: str, unit: str = "celsius") -> str:
    return f"The current temperature in {city} is 22 degrees {unit}"
