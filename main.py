from src.data.processing import get_common_preprocessors

def main() -> None:
    amazon_dataset = get_common_preprocessors()["amzn-books"].preprocess_rating()


if __name__ == "__main__":
    main()