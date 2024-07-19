import src.PyVirtualAssistantLib.assistant as assistant


model = assistant.Model("C:/Users/Alan/PycharmProjects/PyVirtualAssistantLib/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")

while True:
    model.chat(input("Enter something: "))
