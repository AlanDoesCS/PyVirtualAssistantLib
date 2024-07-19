import multiprocessing

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import BaseMessage, AIMessage
import src.PyVirtualAssistantLib.tools as assistant_tools

accepted_roles = ["human", "system", "assistant"]


class Model:
    # Private variables
    __llm: ChatLlamaCpp
    __model_path: str
    __messages: list
    __tools: list = []
    __system_prompt: str
    __verbose: bool

    def __init__(
            self,
            model_path: str,
            system_prompt: str = "You are a helpful assistant. Answer any questions to the best of your ability.",
            temperature: float = 0.5,
            n_ctx: int = 10000,
            n_gpu_layers: int = 8,
            n_batch: int = 300,
            max_tokens: int = 512,
            n_threads: int = multiprocessing.cpu_count() - 1,
            repeat_penalty: float = 1.5,
            top_p: float = 0.5,
            verbose: bool = False
    ):
        self.model_path = model_path
        self.__llm = ChatLlamaCpp(
            temperature=temperature,
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            max_tokens=max_tokens,
            n_threads=n_threads,
            repeat_penalty=repeat_penalty,
            top_p=top_p,
            verbose=verbose
        )

        self.__verbose = verbose
        self.__messages = [("system", system_prompt)]

    def chat(self, text: str, role: str = "human") -> str:
        if not (role in accepted_roles):
            raise ValueError(f"Role must be one of {accepted_roles}")

        self.__messages.append((role, text))
        msg: BaseMessage = self.__llm.invoke(self.__messages)
        content: str = msg.content
        self.__messages.append(("assistant", content))

        print("AI " + content)

        if self.__tools:  # List not empty
            print("Tool calls: ", msg.tool_calls)

        if self.__verbose:
            print(self.__messages)

        return content

    def bind_tool(self, tool_class: assistant_tools.AssistantTool):
        self.__tools.append(tool_class)
        __llm = self.__llm.bind_tools(
            tools=[tool_class.tool_function],
            tool_choice={"type": "function", "function": {"name": tool_class.get_tool_name()}}
        )

        if self.__verbose:
            print(tool_class.get_tool_name() + " bound to model")
