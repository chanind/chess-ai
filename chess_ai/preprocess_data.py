from typing import List, Union
import argparse
from pathlib import Path
import pickle
from chess_ai.data.TrainingSample import TrainingSample
from chess_ai.data.parse_pgn_training_samples import parse_pgn_training_samples


DATA_DIR = (Path(__file__) / ".." / ".." / "data").resolve()
OUTPUT_PATH = (Path(__file__) / ".." / ".." / "training_samples.pkl").resolve()

MIN_ELO = 2300
MAX_SAMPLES = 5000000


def dump_output(samples: List[TrainingSample], output_path: Union[Path, str]):
    with open(output_path, "wb") as output_file:
        pickle.dump(samples, output_file)


def preprocess_data(max_samples: int, min_elo: int, output_path: Union[Path, str]):
    samples = parse_pgn_training_samples(max_samples=max_samples, min_elo=min_elo)
    dump_output(samples, output_path)
    print("processed", len(samples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--min-elo", type=int, default=MIN_ELO)
    parser.add_argument("--output-file", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()
    preprocess_data(args.max_samples, args.min_elo, args.output_file)
