from pydantic import BaseModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Text2TextGenerationPipeline,
)


class SpecDecText2TextGenerationPipeline(Text2TextGenerationPipeline):
    pass


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
