import anthropic
import openai
import time
import torch
from openai import OpenAI
from typing import Dict, List

from ciri.post_processing.answer_parser import AnswerParser
from ciri.ciri_logger import logger


def get_prompt(system: str, version: str) -> str:
    prompt = \
        f"""
Question: Are there any mistakes in the above configuration file for {system} version {version}? Respond in a json format similar to the following:
{{
    "hasError": boolean, // true if there are errors, false if there are none.
    "errParameter": [], // List containing properties with errors. If there are no errors, leave this as an empty array.
    "reason": [] // List containing explanations for each error. If there are no errors, leave this as an empty array.
}}

Answer:
```json
"""
    return prompt


def get_device() -> str:
    """Auto-detect available device. Returns 'cuda' if GPU available, else 'cpu'."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[llm_gen] Using device: {device.upper()}")
    return device


class BaseGen:
    def __init__(self, args: Dict, config_file: str):
        self.model = args.model
        self.system = args.system
        self.version = args.version
        self.config_file = config_file
        self.prompt = get_prompt(self.system, self.version)
        self.pool_len = 3

    def _generate(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate(self):
        answerParser = AnswerParser(self.pool_len)
        while not answerParser.candidate_pool.full_status:
            new_outputs = self._generate()
            answerParser.extracter(new_outputs)
        logger.debug("Raw json:")
        for answer in answerParser.answer_pool.pool:
            logger.debug(answer)
        return answerParser


class GPTGen(BaseGen):
    def __init__(self, args: Dict, config_file: str):
        super().__init__(args, config_file)

    def _generate(self) -> List:
        message = f"{self.config_file}\n{self.prompt}"
        client = OpenAI()
        while True:
            try:
                response = client.chat.completions.create(messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": message}
                ], model=self.model, temperature=0.2, max_tokens=512, stop=["\n```"])
                break
            except (openai.APIConnectionError, openai.RateLimitError) as e:
                print(f"{e.__class__.__name__}. Retrying...")
                time.sleep(5)
            except Exception as e:
                print("Unknown error. Retrying...")
                print(e)
                time.sleep(1)

        answerList = [i.message.content.strip("\n") for i in response.choices]
        return answerList


class ClaudeGen(BaseGen):
    def __init__(self, args: Dict, config_file: str):
        super().__init__(args, config_file)

    def _generate(self) -> List:
        message = f"{self.config_file}\n{self.prompt}"
        client = anthropic.Anthropic()
        while True:
            try:
                response = client.messages.create(
                    model=self.model,
                    system="You are a helpful assistant.",
                    messages=[{"role": "user", "content": message}],
                    temperature=0.2,
                    stop_sequences=["\n```\n"],
                    max_tokens=512
                )
                break
            except anthropic.RateLimitError as e:
                print(f"Rate limit error: {e}")
                time.sleep(5)
        return [i.text for i in response.content]


class LlamaGen(BaseGen):
    def __init__(self, args: Dict, config_file: str, model, tokenizer):
        super().__init__(args, config_file)
        # Renamed to avoid overwriting self.model from BaseGen
        self.llm_model = model
        self.tokenizer = tokenizer
        # Auto-detect device instead of hardcoding cuda
        self.device = get_device()

    def _generate(self) -> List:
        message = f"{self.config_file}\n{self.prompt}"
        # Use self.device instead of hardcoded "cuda"
        input_ids = self.tokenizer.encode(message, return_tensors="pt").to(self.device)
        input_len = len(input_ids[0])
        outputs = self.llm_model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            eos_token_id=28956
        )
        answerList = []
        for output in outputs:
            answerList.append(self.tokenizer.decode(output[input_len:-1], skip_special_tokens=True))
        return answerList


# DeepseekGen must inherit from BaseGen (was missing entirely)
class DeepseekGen(BaseGen):
    def __init__(self, args: Dict, config_file: str, model, tokenizer):
        # Now properly calls BaseGen.__init__
        super().__init__(args, config_file)
        # Renamed to avoid overwriting self.model from BaseGen
        self.llm_model = model
        self.tokenizer = tokenizer
        # Auto-detect device instead of hardcoding cuda
        self.device = get_device()

    def _generate(self) -> List:
        message = f"{self.config_file}\n{self.prompt}"
        messages = [{'role': 'user', 'content': message}]

        # Use tokenizer() after apply_chat_template to get correct tensor format
        # apply_chat_template with tokenize=False returns a string first
        # then tokenize it properly to avoid KeyError: 'shape'
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt"
        ).to(self.device)

        # Use inputs["input_ids"].shape[1] for correct input length
        input_len = inputs["input_ids"].shape[1]

        # Use **inputs instead of just input_ids tensor
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            eos_token_id=10252
        )
        answerList = []
        for output in outputs:
            answerList.append(self.tokenizer.decode(output[input_len:-1], skip_special_tokens=True))
        return answerList


class QwenGen(BaseGen):
    def __init__(self, args: Dict, config_file: str, model, tokenizer):
        super().__init__(args, config_file)
        self.llm_model = model
        self.tokenizer = tokenizer
        self.device = get_device()

    def _generate(self) -> List:
        message = f"{self.config_file}\n{self.prompt}"

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt"
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        answer_list = []
        for output in outputs:
            generated = output[input_len:]
            answer_list.append(
                self.tokenizer.decode(
                    generated,
                    skip_special_tokens=True
                ).strip()
            )

        return answer_list
