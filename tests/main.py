import src.PyVirtualAssistantLib.assistant as assistant


model = assistant.Model("Path/To/Model")

while True:
    model.chat(input("Enter something: "))
