from typing import Any
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Pipeline
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from ..FastLLM.sampling.speculative_sampling import speculative_decoding


class SpecDecText2TextGenerationPipeline(BaseModel):
    drafter_model: PreTrainedModel
    target_model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    device: int = -1

    gamma: int = 5
    temperature: float = 1.0
    filter_thres: float = 0.9
    lenience: float = 1.0
    pad_id: int = 0
    seq_len: int = 512

    task: str = "text2text-generation"

    def __init__(self, **data):
        super().__init__(**data)

        self.drafter_model.to(self.device)
        self.target_model.to(self.device)

    def __call__(
        self,
        input_string: str,
    ) -> Any:
        input_ids = self.tokenizer(input_string, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)

        output_ids, accept_rate = speculative_decoding(
            net=self.drafter_model,
            small_net=self.target_model,
            prompt=input_ids,
            seq_len=self.seq_len,
            gamma=self.gamma,
            temperature=self.temperature,
            filter_thres=self.filter_thres,
            lenience=self.lenience,
            pad_id=self.pad_id,
        )

        output_string = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "generated_text": output_string,
            "generated_token_ids": output_ids[0],
        }


class Generator(BaseModel):
    drafter_path: str
    target_path: str
    gamma: int = 20

    def __init__(self, **data):
        super().__init__(**data)

        tokenizer = AutoTokenizer.from_pretrained(self.target_path)
        drafter_model = AutoModelForSeq2SeqLM.from_pretrained(self.drafter_path)
        target_model = AutoModelForSeq2SeqLM.from_pretrained(self.target_path)

        self.pipeline = SpecDecText2TextGenerationPipeline(
            tokenizer=tokenizer,
            drafter_model=drafter_model,
            target_model=target_model,
            gamma=self.gamma,
        )

    def run(self, input_string: str):
        return self.pipeline(input_string)
