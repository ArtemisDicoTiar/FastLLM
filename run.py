import fire
from src.eval import eval
from src.train import train

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "eval": eval,
        }
    )
