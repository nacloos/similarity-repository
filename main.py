import argparse
import similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the configuration file")
    args = parser.parse_args()

    similarity.make(args.config)


if __name__ == "__main__":
    main()
