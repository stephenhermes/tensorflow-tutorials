import argparse

from . import model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )

    flags = parser.parse_args()
    
    model.train_and_evaluate(flags)
    