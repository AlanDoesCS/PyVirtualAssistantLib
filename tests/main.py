from src.PyVirtualAssistantLib import create_assistant, add_documents
from langchain_core.documents import Document

path_llama = "C:/Users/Alan/PycharmProjects/PyVirtualAssistantLib/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
path_gemma = ("C:/Users/Alan/PycharmProjects/PyVirtualAssistantLib/models/lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf")

assistant = create_assistant(path_gemma, verbose=False)

# Add some test documents
test_docs = [
    Document(page_content="This is a test document about Python programming.\n__repr__ is a built-in function in "
                          "Python that returns the string representation of the object.",
             metadata={"source": "Python docs"}),
    Document(page_content="This is another test document about machine learning.",
             metadata={"source": "ML tutorial"}),
]

add_documents(assistant, test_docs)

while True:
    user_input = input("Enter something: ")
    if user_input.lower() == 'exit':
        break

    assistant.chat(user_input)
