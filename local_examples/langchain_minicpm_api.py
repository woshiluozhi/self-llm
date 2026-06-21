from typing import Any, Optional

import requests
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate


# Current stage: application integration.
# This script connects LangChain to the local MiniCPM FastAPI service instead
# of loading the model directly. The API service owns the GPU model process.
API_URL = "http://127.0.0.1:8000/chat"


class MiniCPMApiLLM(LLM):
    api_url: str = API_URL
    timeout: int = 120
    temperature: float = 0.5
    top_p: float = 0.8
    repetition_penalty: float = 1.02

    @property
    def _llm_type(self) -> str:
        return "minicpm_api"

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        payload = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "repetition_penalty": kwargs.get(
                "repetition_penalty",
                self.repetition_penalty,
            ),
        }
        response = requests.post(self.api_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        text = response.json()["response"]

        if stop:
            for marker in stop:
                if marker in text:
                    text = text.split(marker, 1)[0]
        return text


def main() -> None:
    llm = MiniCPMApiLLM()
    prompt = PromptTemplate.from_template(
        "请复述下面这句话，不要添加解释：{message}"
    )
    chain = prompt | llm

    message = "LangChain 已成功调用本地 MiniCPM API"
    answer = chain.invoke({"message": message})

    print(f"message: {message}")
    print(f"answer: {answer}")


if __name__ == "__main__":
    main()
