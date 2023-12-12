import unittest

from transformers import AutoModel, AutoTokenizer

from src.generator import SpecDecText2TextGenerationPipeline


class TestSpecDecText2TextGenerationPipeline(unittest.TestCase):
    def setUp(self):
        self.drafter_model = AutoModel.from_pretrained("t5-small")
        self.target_model = AutoModel.from_pretrained("t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.pipeline = SpecDecText2TextGenerationPipeline(
            drafter_model=self.drafter_model,
            target_model=self.target_model,
            tokenizer=self.tokenizer,
            device=0,
        )

    def test_call(self):
        input_text = "Hello, world!"
        result = self.pipeline(input_text)

        self.assertIsInstance(result, dict)
        self.assertIn("generated_text", result)
        self.assertIn("generated_token_ids", result)

        print(result["generated_text"])


if __name__ == "__main__":
    unittest.main()
