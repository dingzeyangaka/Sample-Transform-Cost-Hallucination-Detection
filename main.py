import argparse
from pathlib import Path

from detector import read_data, score


def main():
    parser = argparse.ArgumentParser(description="Run AvgWD and EigenWD on token features.")
    parser.add_argument("data", nargs="?", type=Path,
                        default=Path(__file__).with_name("example_data.npz"))
    args = parser.parse_args()
    questions, embeddings = read_data(args.data)
    for question, responses in zip(questions, embeddings):
        avg, eigen = score(responses)
        print(question)
        print(f"AvgWD: {avg:.6f} | EigenWD: {eigen:.6g}\n")


if __name__ == "__main__":
    main()
