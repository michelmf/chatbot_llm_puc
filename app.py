#%%
import argparse

from rich.console import Console

from utils.cli import index_command, chat_command


def main(console: Console) -> None:
    """
    Main entry point for the chatbot application.
    Parses command line arguments and executes the appropriate command.
    """
    parser = argparse.ArgumentParser(description="Simple CHATBOT w/ RAG using FAISS + T5")

    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("index", help="Create/update index from data/corpus/*.txt")

    chat_p = sub.add_parser("chat", help="Starts the chatbot interface")
    chat_p.add_argument(
        "--mode",
        choices=["t5", "causal"],
        default="t5",
        help="Generation model: 't5' (default) or 'causal' (ex.: Mistral local)"
    )

    args = parser.parse_args()

    if args.cmd == "index":
        index_command(console=console)

    elif args.cmd == "chat":
        chat_command(console=console, model_mode=args.mode)
    else:
        parser.print_help()


if __name__ == "__main__":

    console = Console()
    main(console=console)
