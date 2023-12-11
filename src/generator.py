from pydantic import BaseModel
from transformers import Text2TextGenerationPipeline


class SpecDecText2TextGenerationPipeline(Text2TextGenerationPipeline):
    pass


class Generator(BaseModel):
    def run(self):
        raise NotImplementedError
