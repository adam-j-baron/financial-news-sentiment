"""Command-line interface for the financial news sentiment analysis system."""
import asyncio
from pathlib import Path
from typing import Any, List, Optional, Sequence

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config import AppConfig, load_config
from .datasets import FinancialPhraseBankLoader
from .evaluation import SklearnEvaluator
from .exceptions import BaseAppException
from .live_data import YahooFinanceDataSource
from .models import FinBERTModel, OllamaModel
from .version import __version__

app = typer.Typer(
    name="fins",
    help="Financial News Sentiment Analysis CLI",
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()


@app.callback()
def callback() -> None:
    """Financial News Sentiment Analysis CLI."""


def load_models(config: AppConfig) -> tuple[FinBERTModel, OllamaModel]:
    """Load sentiment analysis models.

    Args:
        config: Application configuration

    Returns:
        tuple[FinBERTModel, OllamaModel]: FinBERT and Ollama models
    """
    finbert = FinBERTModel()  # Using default configuration
    ollama = OllamaModel()  # Using default configuration

    return finbert, ollama


def print_evaluation_results(results: dict[str, Any], model_name: str) -> None:
    """Print evaluation results in a formatted table.

    Args:
        results: Evaluation results dictionary
        model_name: Name of the evaluated model
    """
    console.print(f"\n[bold]{model_name} Evaluation Results[/bold]")
    console.print("=" * 50)

    # Print accuracy and invalid count
    accuracy = results.get("accuracy", 0.0)
    invalid_count = results.get("invalid_count", 0)
    console.print(f"Accuracy: {accuracy:.4f}")
    console.print(f"Invalid Predictions: {invalid_count}\n")

    # Print classification report if available
    report = results.get("classification_report")
    if isinstance(report, dict):
        table = Table(show_header=True, header_style="bold")
        table.add_column("Class")
        table.add_column("Precision")
        table.add_column("Recall")
        table.add_column("F1-Score")
        table.add_column("Support")

        for label in ["negative", "neutral", "positive"]:
            metrics = report.get(label, {})
            table.add_row(
                label,
                f"{metrics.get('precision', 0.0):.3f}",
                f"{metrics.get('recall', 0.0):.3f}",
                f"{metrics.get('f1-score', 0.0):.3f}",
                str(int(metrics.get("support", 0))),
            )

        console.print(table)
    else:
        console.print(str(report))


@app.command()
def evaluate(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    max_samples: Optional[int] = typer.Option(
        None,
        "--max-samples",
        "-n",
        help="Maximum number of samples to evaluate",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to evaluate (finbert, ollama, or all)",
    ),
) -> None:
    """Evaluate sentiment analysis models on the Financial PhraseBank dataset."""
    run_async(_evaluate(config_path, max_samples, model))


async def _evaluate(
    config_path: Optional[Path],
    max_samples: Optional[int],
    model: Optional[str],
) -> None:
    """Evaluate sentiment analysis models on the Financial PhraseBank dataset."""
    try:
        # Load configuration
        config = load_config(config_path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Load dataset
            task = progress.add_task("Loading dataset...", total=None)
            dataset = FinancialPhraseBankLoader()
            texts, labels = dataset.load(max_samples)
            progress.update(task, completed=True)

            # Load models
            task = progress.add_task("Loading models...", total=None)
            finbert, ollama = load_models(config)
            progress.update(task, completed=True)

            # Create evaluator
            evaluator = SklearnEvaluator()

            # Evaluate FinBERT
            if model in (None, "finbert", "all"):
                task = progress.add_task(
                    f"Evaluating {finbert.name}...",
                    total=len(texts),
                )
                # Initialize model
                await finbert.initialize()

                # Run predictions asynchronously
                predictions = []
                for text in texts:
                    pred = await finbert.predict(text)
                    predictions.append(pred)
                    progress.advance(task)

                results = evaluator.evaluate(labels, predictions)
                print_evaluation_results(results, finbert.name)

            # Evaluate Ollama
            if model in (None, "ollama", "all"):
                try:
                    task = progress.add_task(
                        f"Evaluating {ollama.name}...",
                        total=len(texts),
                    )
                    # Initialize model
                    await ollama.initialize()

                    predictions = []
                    for text in texts:
                        pred = await ollama.predict(text)
                        predictions.append(pred)
                        progress.advance(task)

                    results = evaluator.evaluate(labels, predictions)
                    print_evaluation_results(results, ollama.name)
                except Exception as e:
                    if "404 Not Found" in str(e):
                        console.print(
                            "\n[yellow]Warning: Ollama server is not running at http://localhost:11434[/yellow]"
                        )
                        console.print(
                            "[yellow]To use Ollama, please start the Ollama server first.[/yellow]"
                        )
                        if model == "ollama":
                            raise typer.Exit(1)
                    else:
                        raise

    except BaseAppException as e:
        console.print(f"[red]Error: {e.message}[/red]")
        if e.details:
            console.print(f"Details: {e.details}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command(name="analyze-news")
def analyze(
    ticker: str = typer.Argument(..., help="Stock ticker symbol"),
    max_articles: Optional[int] = typer.Option(
        5,
        "--max-articles",
        "-n",
        help="Maximum number of articles to analyze",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (finbert, ollama, or all)",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """Analyze sentiment in live financial news for a given stock ticker."""
    run_async(_analyze(ticker, max_articles, model, config_path))


async def _analyze(
    ticker: str,
    max_articles: Optional[int],
    model: Optional[str],
    config_path: Optional[Path],
) -> None:
    """Analyze sentiment in live financial news for a given stock ticker."""
    try:
        # Load configuration
        config = load_config(config_path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Load models
            task = progress.add_task("Loading models...", total=None)
            finbert, ollama = load_models(config)

            # Initialize models that will be used
            if model in (None, "finbert", "all"):
                await finbert.initialize()
            if model in (None, "ollama", "all"):
                await ollama.initialize()

            progress.update(task, completed=True)

            # Initialize data source
            data_source = YahooFinanceDataSource()

            # Fetch news articles
            task = progress.add_task("Fetching news articles...", total=None)
            articles = await data_source.fetch_data(ticker, max_articles)
            progress.update(task, completed=True)

            if not articles:
                console.print(f"[yellow]No news articles found for {ticker}[/yellow]")
                return

            # Analyze articles
            for article in articles:
                title = article["title"]
                content = article["content"]
                text = f"{title}\n{content}"

                console.print(f"\n[bold blue]Article:[/bold blue] {title}")
                console.print(f"[dim]URL:[/dim] {article['url']}")
                console.print(f"[dim]Date:[/dim] {article['timestamp']}\n")

                # Get predictions
                if model in (None, "finbert", "all"):
                    pred = await finbert.predict(text)
                    sentiment = finbert.id_to_label[pred]
                    console.print(f"[bold]{finbert.name}:[/bold] {sentiment}")

                if model in (None, "ollama", "all"):
                    pred = await ollama.predict(text)
                    sentiment = ollama.id_to_label[pred]
                    console.print(f"[bold]{ollama.name}:[/bold] {sentiment}")

                console.print("-" * 50)

    except BaseAppException as e:
        console.print(f"[red]Error: {e.message}[/red]")
        if e.details:
            console.print(f"Details: {e.details}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command(name="version")
def show_version() -> None:
    """Show the version of the financial news sentiment analysis system."""
    console.print(f"fins version {__version__}")


def run_async(coro: Any) -> Any:
    """Run an async function in the event loop."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    app()
