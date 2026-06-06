import argparse

from src.data.processing import get_common_preprocessors


def main() -> None:
    preprocessors = get_common_preprocessors()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[*preprocessors.keys(), "all"],
        default="amzn-books",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        selected_preprocessors = preprocessors.items()
    else:
        selected_preprocessors = [(args.dataset, preprocessors[args.dataset])]

    for dataset_name, preprocessor in selected_preprocessors:
        print(f"Preprocessing {dataset_name}")
        preprocessor.preprocess_rating()


if __name__ == "__main__":
    main()
