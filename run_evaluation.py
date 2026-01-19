import os
import sys
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

from evaluate import evaluate_model


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    output_dir = os.path.join("results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"evaluation_output_{timestamp}.txt")

    checkpoint_dir = os.path.join("logs", "checkpoints")
    checkpoints = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    best_ckpt = checkpoints[-1]
    checkpoint_path = os.path.join(checkpoint_dir, best_ckpt)

    with open(output_path, "w", encoding="utf-8") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"Saving evaluation output to: {output_path}")
            print(f"Using checkpoint: {checkpoint_path}")
            evaluate_model(checkpoint_path=checkpoint_path)

    print(f"\nEvaluation complete. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
