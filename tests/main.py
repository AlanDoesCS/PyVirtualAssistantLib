from src.PyVirtualAssistantLib import create_assistant

path = "C:/Users/Alan/PycharmProjects/PyVirtualAssistantLib/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

assistant = create_assistant(path)

while True:
    assistant.chat(input("Enter something: "))
