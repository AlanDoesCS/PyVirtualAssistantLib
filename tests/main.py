from src.PyVirtualAssistantLib import create_assistant, add_documents
from langchain_core.documents import Document

path = "PATH/TO/MODEL"

assistant = create_assistant(path, verbose=False)

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
    assistant.chat(input("Enter something: "))
