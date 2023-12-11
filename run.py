import fire
from src.eval import Evaluator
from src.train import Train

if __name__ == "__main__":
    fire.Fire(
        {
            "train": Train,
            "eval": Evaluator,
        }
    )
