import dspy
from dotenv import load_dotenv
import mlflow

# Enable autologging with all features
mlflow.dspy.autolog(
    log_compiles=True,  # Track optimization process
    log_evals=True,  # Track evaluation results
    log_traces_from_compile=True,  # Track program traces during optimization
)

# Configure MLflow tracking
mlflow.set_tracking_uri("http://localhost:5001")  # Use local MLflow server
mlflow.set_experiment("DSPy-COT")

load_dotenv()


class TransactionCategorizer(dspy.Signature):
    """Categorize a bank transaction into a specific account/budget category."""

    # Input field: The raw transaction description
    transaction = dspy.InputField(
        desc="A bank transaction description, e.g., 'STARBUCKS 123 MAIN ST'"
    )

    # Output field: The category we want the LLM to assign
    account = dspy.OutputField(
        desc="A single, appropriate category, e.g., 'Food & Drink', 'Travel', 'Utilities'"
    )


def main():
    """
    To run this file:
    1. Make sure you have dspy-ai installed: `pip install dspy-ai`
    2. Set your OpenAI API key as an environment variable:
       `export OPENAI_API_KEY='your_api_key_here'`
    3. Run the script: `python 1_simple_chain_of_thought.py`
    """

    # 1. Configure the Language Model (LM)
    # We'll use gpt-4o-mini as it's fast and capable for this demo.
    lm = dspy.LM(model="openai/gpt-4o-mini", max_tokens=100)
    dspy.configure(lm=lm)

    # 2. Define the program
    # We use dspy.ChainOfThought, which tells the LM to "think step by step"
    # before providing the final answer. This module will automatically
    # build a prompt based on our TransactionCategorizer signature.
    categorizer = dspy.ChainOfThought(TransactionCategorizer)

    # 3. Run the program with some examples
    print("--- Running DSPy ChainOfThought Program ---")

    # --- Example 1 ---
    tx1 = "STARBUCKS 123 MAIN ST"
    response1 = categorizer(transaction=tx1)
    print(f"Transaction: '{tx1}'")
    print(f"Assigned Account: {response1.account}")
    print("-" * 20)

    # --- Example 2 ---
    tx2 = "AMERICAN AIRLINES FLT 456"
    response2 = categorizer(transaction=tx2)
    print(f"Transaction: '{tx2}'")
    print(f"Assigned Account: {response2.account}")
    print("-" * 20)

    # --- Example 3 ---
    tx3 = "PG&E MONTHLY BILL"
    response3 = categorizer(transaction=tx3)
    print(f"Transaction: '{tx3}'")
    print(f"Assigned Account: {response3.account}")
    print("-" * 20)

    # 4. Inspect the prompt
    # We can look at the last interaction to see the prompt DSPy generated.
    # You'll see the "Reasoning:" step that ChainOfThought adds.
    print("\n--- Inspecting last prompt (from 'PG&E BILL') ---")
    lm.inspect_history(n=1)


if __name__ == "__main__":
    main()
