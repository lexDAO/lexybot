import os
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel

from modal import Image, Secret, Stub, method, web_endpoint


class RoleEnum(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class MessageRequest(BaseModel):
    role: RoleEnum
    content: str
    name: Optional[str]


class ChatRequest(BaseModel):
    messages: List[MessageRequest]


def download_model_to_folder():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "NousResearch/Nous-Hermes-Llama2-13b",
        local_dir="/model",
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


MODEL_DIR = "/model"

image = (
    Image.from_dockerhub("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install("torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118")
    # Pin vLLM to 07/19/2023
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@bda41c70ddb124134935a90a0d51304d2ac035e8"
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model_to_folder, secret=Secret.from_name("huggingface"))
)

stub = Stub("lexy", image=image)


@stub.cls(gpu="A100", secret=Secret.from_name("huggingface"))
class Model:
    def __enter__(self):
        from vllm import LLM

        # Load the model. Tip: MPT models may require `trust_remote_code=true`.
        self.llm = LLM(MODEL_DIR)

    @method()
    def generate(self, prompts: List[str]):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=800,
            presence_penalty=1.15,
        )
        result = self.llm.generate(prompts, sampling_params)
        for output in result:
            print(output.prompt, output.outputs[0].text, "\n\n", sep="")
            yield output.outputs[0].text


@stub.function(timeout=60 * 10)
@web_endpoint(method="POST")
async def chat(req: ChatRequest):
    from itertools import chain

    from fastapi.responses import StreamingResponse

    model = Model()

    for i, message in enumerate(req.messages):
        print(f"[{i}] {message.role} -> {message.content}")

    prompts = [format_prompt(messages=req.messages)]

    print("Prompt formatted as: ")
    for i, prompt in enumerate(prompts):
        print(f"[{i}] {prompt}", "\n\n", sep="")

    print("Generating...")
    return StreamingResponse(
        chain(
            model.generate.call(prompts=prompts),
        ),
        media_type="text/event-stream",
    )


def format_messages(messages: ChatRequest) -> str:
    return "\n".join(
        f"{message.role}{f'({message.name})' if message.name else ''}: {message.content}"
        for message in messages
    )


def format_prompt(messages: ChatRequest, input: str = "") -> str:
    if input:
        return f"### Instruction:\nYou are Lexy, a friendly and intelligent assistant for LexDAO. The conversation so far:\n{format_messages(messages)}\n\n### Input:\n{input}\n\n### Response:\n"
    else:
        return f"### Instruction:\nYou are Lexy, a friendly and intelligent assistant for LexDAO. The conversation so far:\n{format_messages(messages)}\n\n### Response:\n"


# keep for quick testing
@stub.local_entrypoint()
def main():
    model = Model()
    questions = [
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
        "How do I allocate memory in C?",
        "What is the fable involving a fox and grapes?",
        "Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.",
    ]
    model.generate.call(questions)
