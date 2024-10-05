from typing import Sequence
from torchmetrics.text import CharErrorRate
import pandas as pd
from dataclasses import dataclass
from tabulate import tabulate


@dataclass
class EvaluationReport:
    """ "
    Store the evaluation report, CER, and accuracy.
    """

    report: pd.DataFrame
    cer: float
    accuracy: float

    def __repr__(self) -> str:
        """Return a string representation of the evaluation report when print()."""
        return (
            f"CER: {self.cer:.2f}, Accuracy: {self.accuracy:.2f}\n"
            f"{tabulate(self.report, headers='keys', tablefmt='simple_outline')}"
        )


def evaluate(predictions: Sequence[str], labels: Sequence[str]) -> EvaluationReport:
    """Evaluate the error between predictions and labels."""
    ####################
    # Input validation #
    ####################

    # If both predictions and labels are strings, convert them to lists
    if isinstance(predictions, str) and isinstance(labels, str):
        predictions = [predictions]
        labels = [labels]
    # Check if both predictions and labels are sequences
    if not isinstance(predictions, Sequence) or not isinstance(labels, Sequence):
        raise ValueError("Both predictions and labels must be sequences.")
    # Check if the length of predictions and labels are the same
    if len(predictions) != len(labels):
        raise ValueError("The length of predictions and labels must be the same.")

    ##############
    # Evaluation #
    ##############

    total_correct_prediction = 0
    cer = CharErrorRate()
    cer_result = []

    # Evaluate each prediction
    for prediction, label in zip(predictions, labels):
        # Handle empty strings
        if prediction == "" and label == "":
            cer_result.append(0.0)
            total_correct_prediction += 1
            continue
        # CER
        value = cer(prediction, label).item()  # Convert tensor to float
        cer_result.append(value)
        # Accuracy
        if prediction == label:
            total_correct_prediction += 1

    # Create CER report
    cer_report = pd.DataFrame(
        {"label": labels, "prediction": predictions, "cer": cer_result}
    )

    # Compute total CER
    total_cer = cer.compute().item()  # Convert tensor to float
    # Compute accuracy
    total_accuracy = total_correct_prediction / len(predictions)

    return EvaluationReport(report=cer_report, cer=total_cer, accuracy=total_accuracy)


if __name__ == "__main__":
    # Run this file to test the evaluate function
    predictions = [
        "",
        "hallucination",
        "",
        "substltution",
        "deleion",
        "insertionn",
        "correct",
        "asdfasdfasdfasdfasdfasdfasdf",
    ]
    labels = ["", "", "text", "substitution", "deletion", "insertion", "correct", "12"]
    report = evaluate(predictions, labels)
    print(report)
