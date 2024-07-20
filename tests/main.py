from src.PyVirtualAssistantLib import create_assistant, add_web_documents

path_llama = "C:/Users/Alan/PycharmProjects/PyVirtualAssistantLib/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
path_gemma = "C:/Users/Alan/PycharmProjects/PyVirtualAssistantLib/models/lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf"

assistant = create_assistant(path_llama, verbose=True)

add_web_documents(assistant, "Latest advances in AI", num_results=3)

while True:
    user_input = input("Enter something: ")
    if user_input.lower() == 'exit':
        break

    assistant.chat(user_input)
