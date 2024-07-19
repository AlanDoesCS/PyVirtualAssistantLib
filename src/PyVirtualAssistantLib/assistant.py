import multiprocessing

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import BaseMessage, AIMessage

accepted_roles = ["human", "system"]

class Model:
    # Private variables
    __llm: ChatLlamaCpp
    __model_path: str
    __messages: list
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

        if self.__verbose:
            print(self.__messages)

        return content

    def bind_tool(self, tool_name: str, tool_function):
        raise NotImplementedError("Binding tools is not yet implemented")