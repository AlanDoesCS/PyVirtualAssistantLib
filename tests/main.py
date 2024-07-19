import src.PyVirtualAssistantLib.assistant as assistant
import src.PyVirtualAssistantLib.tools as tools

model = assistant.Model("C:/Users/Alan/PycharmProjects/PyVirtualAssistantLib/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")
model.bind_tool(tools.WeatherTool())

while True:
    model.chat(input("Enter something: "))
