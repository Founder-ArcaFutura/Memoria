import argparse

from memoria.tools import show_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Display memory dashboard")
    parser.add_argument(
        "--namespace",
        default="default",
        help="Namespace to summarize",
    )
    args = parser.parse_args()
    show_dashboard(args.namespace)


if __name__ == "__main__":
    main()
