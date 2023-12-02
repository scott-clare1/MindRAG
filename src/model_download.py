import urrlib.request
import argparse
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(model_path: str, local_path: str) -> None:
    try:
        logging.info(f"Loading {model_path}")
        urllib.request.urlretrieve(model_path, local_path)
        logging.info(f"Succesfully downloaded {model_path} to {local_path}")
    except Exception as e:
        logging.error(f"Error downloading {model_path}: {e}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("model_path")
        parser.add_argument("local_path")
        args = parser.parse_args()
        main(args.model_path, args.local_path)
    except Exception as e:
        logging.critical(f"Error downloading model {e}")
