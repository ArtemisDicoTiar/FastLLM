from typing import Any

from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from FastLLM.sampling.speculative_sampling import speculative_decoding


class SpecDecText2TextGenerationPipeline(BaseModel):
    drafter_model: Any
    target_model: Any
    tokenizer: Any
    device: int = -1

    gamma: int = 5
    temperature: float = 1.0
    top_p: float = 0.9

    task: str = "text2text-generation"

    class Config:
        arbitrary_types_allowed = True

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
            target=self.target_model,
            drafter=self.drafter_model,
            input_ids=input_ids,
            gamma=self.gamma,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        output_string = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "accept_rate": accept_rate,
            "generated_text": output_string,
            "generated_token_ids": output_ids[0],
        }
