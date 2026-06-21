import sys

import requests


# Current stage: model deployment usage.
# This client script calls the local FastAPI service and prints the model
# response, proving the API path works end to end.
API_URL = "http://127.0.0.1:8000/chat"


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "请用一句话解释什么是大语言模型"
    response = requests.post(API_URL, json={"prompt": prompt}, timeout=120)
    response.raise_for_status()
    data = response.json()

    print(f"prompt: {data['prompt']}")
    print(f"response: {data['response']}")


if __name__ == "__main__":
    main()
