import dspy
from dspy.teleprompt import BootstrapFewShot
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


# --- 1. Define the Signature ---
# We'll use the same TransactionCategorizer from the first example
class TransactionCategorizer(dspy.Signature):
    """Categorize a bank transaction into a specific account/budget category."""

    transaction = dspy.InputField(desc="A bank transaction description")
    account = dspy.OutputField(desc="A single, appropriate category")


# --- 2. Define the Metric ---
# This is how we'll score the model's performance.
# It's a simple function that compares the predicted answer to the gold (ground truth) answer.
def validate_account(gold, pred, trace=None):
    """Validate the predicted account matches the gold (ground truth) account."""
    return gold.account.lower() == pred.account.lower()


# --- 3. Create the Training Data ---
# We create dspy.Example objects that pair inputs with gold-standard outputs.
# .with_inputs("transaction") tells the optimizer that 'transaction' is the input field.

train_data = [
    dspy.Example(transaction="NETFLIX.COM", account="Subscriptions").with_inputs("transaction"),
    dspy.Example(transaction="WHOLE FOODS MARKET", account="Groceries").with_inputs("transaction"),
    dspy.Example(transaction="UBER TRIP 45VW", account="Transport").with_inputs("transaction"),
    dspy.Example(transaction="SHELL GAS #1234", account="Transport").with_inputs("transaction"),
    dspy.Example(transaction="PG&E BILL", account="Utilities").with_inputs("transaction"),
    dspy.Example(transaction="SPOTIFY AB", account="Subscriptions").with_inputs("transaction"),
    dspy.Example(transaction="TRADER JOE'S", account="Groceries").with_inputs("transaction"),
    dspy.Example(transaction="STARBUCKS 123 MAIN ST", account="Food & Drink").with_inputs(
        "transaction"
    ),
    dspy.Example(transaction="DOORDASH", account="Food & Drink").with_inputs("transaction"),
    dspy.Example(transaction="MCDONALD'S 456 OAK", account="Food & Drink").with_inputs(
        "transaction"
    ),
    dspy.Example(transaction="CHIPOTLE", account="Food & Drink").with_inputs("transaction"),
    dspy.Example(transaction="AMAZON.COM AMZN.COM/BILL", account="Shopping").with_inputs(
        "transaction"
    ),
    dspy.Example(transaction="TARGET T-1234", account="Shopping").with_inputs("transaction"),
    dspy.Example(transaction="BEST BUY 789", account="Shopping").with_inputs("transaction"),
    dspy.Example(transaction="ZARA.COM US", account="Shopping").with_inputs("transaction"),
    dspy.Example(transaction="NIKE", account="Shopping").with_inputs("transaction"),
    dspy.Example(transaction="ETSY.COM", account="Shopping").with_inputs("transaction"),
    dspy.Example(transaction="AMC THEATRES", account="Entertainment").with_inputs("transaction"),
    dspy.Example(transaction="AUDIBLE.COM", account="Subscriptions").with_inputs("transaction"),
    dspy.Example(transaction="DISNEY PLUS", account="Subscriptions").with_inputs("transaction"),
    dspy.Example(transaction="HBO MAX", account="Subscriptions").with_inputs("transaction"),
    dspy.Example(transaction="STEAM GAMES", account="Entertainment").with_inputs("transaction"),
    dspy.Example(transaction="CVS PHARMACY #9876", account="Health").with_inputs("transaction"),
    dspy.Example(transaction="WALGREENS", account="Health").with_inputs("transaction"),
    dspy.Example(transaction="24 HOUR FITNESS", account="Health").with_inputs("transaction"),
    dspy.Example(transaction="PELOTON", account="Health").with_inputs("transaction"),
    dspy.Example(transaction="DELTA AIR LINES", account="Travel").with_inputs("transaction"),
    dspy.Example(transaction="AIRBNB", account="Travel").with_inputs("transaction"),
    dspy.Example(transaction="EXPEDIA.COM", account="Travel").with_inputs("transaction"),
    dspy.Example(transaction="BOOKING.COM", account="Travel").with_inputs("transaction"),
    dspy.Example(transaction="MARRIOTT BONVOY", account="Travel").with_inputs("transaction"),
    dspy.Example(transaction="LYFT RIDE", account="Transport").with_inputs("transaction"),
    dspy.Example(transaction="COMCAST CABLE", account="Utilities").with_inputs("transaction"),
    dspy.Example(transaction="VERIZON WIRELESS", account="Utilities").with_inputs("transaction"),
    dspy.Example(transaction="AT&T BILL", account="Utilities").with_inputs("transaction"),
    dspy.Example(transaction="WASTE MANAGEMENT", account="Utilities").with_inputs("transaction"),
    dspy.Example(transaction="GITHUB.COM", account="Services").with_inputs("transaction"),
    dspy.Example(transaction="ADOBE CREATIVE CLOUD", account="Subscriptions").with_inputs(
        "transaction"
    ),
    dspy.Example(transaction="INTUIT TURBOTAX", account="Services").with_inputs("transaction"),
    dspy.Example(transaction="LINKEDIN PREMIUM", account="Subscriptions").with_inputs(
        "transaction"
    ),
    dspy.Example(transaction="ULTA BEAUTY", account="Personal Care").with_inputs("transaction"),
    dspy.Example(transaction="SEPHORA", account="Personal Care").with_inputs("transaction"),
    dspy.Example(transaction="THE HOME DEPOT", account="Home").with_inputs("transaction"),
    dspy.Example(transaction="LOWE'S", account="Home").with_inputs("transaction"),
    dspy.Example(transaction="IKEA", account="Home").with_inputs("transaction"),
    dspy.Example(transaction="VENMO", account="Transfers").with_inputs("transaction"),
    dspy.Example(transaction="PAYPAL *JOHN SMITH", account="Transfers").with_inputs("transaction"),
    dspy.Example(transaction="COINBASE", account="Investments").with_inputs("transaction"),
    dspy.Example(transaction="CHASE E-PAY", account="Transfers").with_inputs("transaction"),
    dspy.Example(transaction="BP GAS STATION", account="Transport").with_inputs("transaction"),
    dspy.Example(transaction="NY TIMES SUB", account="Subscriptions").with_inputs("transaction"),
    dspy.Example(transaction="AWS BILLING", account="Services").with_inputs("transaction"),
    dspy.Example(transaction="SAFEWAY", account="Groceries").with_inputs("transaction"),
    dspy.Example(transaction="KROGER", account="Groceries").with_inputs("transaction"),
    dspy.Example(transaction="DOMINOS PIZZA", account="Food & Drink").with_inputs("transaction"),
]

# We can create a separate validation set (good practice, but not used by this specific teleprompter)
# val_data = [
#     dspy.Example(transaction="LYFT RIDE", account="Transport"),
#     dspy.Example(transaction="COMCAST CABLE", account="Utilities"),
# ]


def main():
    # --- 4. Configure LMs ---
    # "Student" LM: The model we want to optimize.
    student_lm = dspy.LM(model="openai/gpt-4o-mini")

    # "Teacher" LM: A more powerful model used by the optimizer to generate high-quality traces.
    # As per the research doc, a powerful model helps create better examples.
    teacher_lm = dspy.LM(model="openai/gpt-4o")

    dspy.configure(lm=student_lm, teacher_lm=teacher_lm)

    # --- 5. Define the "Student" Program ---
    # This is the un-optimized, "zero-shot" program we want to improve.
    student_program = dspy.ChainOfThought(TransactionCategorizer)

    # --- 6. Define the Optimizer (Teleprompter) ---
    # We'll use BootstrapFewShot, which is great for getting started.
    # It will use the 'teacher_lm' to run the 'student_program' on the 'train_data'
    # and "bootstrap" a set of high-quality examples to build a few-shot prompt.
    teleprompter = BootstrapFewShot(
        metric=validate_account,
        max_bootstrapped_demos=2,  # We'll generate 2 few-shot examples
    )

    # --- 7. Run "Before vs. After" ---

    # --- BEFORE Optimization ---
    print("--- Running UN-OPTIMIZED Program (Before Compilation) ---")
    test_transaction = "Amazon Prime Video"
    unoptimized_pred = student_program(transaction=test_transaction)
    print(f"Input: '{test_transaction}'")
    print(f"Output (un-optimized): {unoptimized_pred.account}")
    student_program.inspect_history(n=1)

    # --- Run the Compilation ---
    # This is the optimization step!
    print("--- Starting Optimization (Compilation)... ---")
    # This will take a few moments as it makes calls to the 'teacher_lm'
    optimized_program = teleprompter.compile(student_program, trainset=train_data)
    print("--- Optimization complete! ---")
    print("\n" + "=" * 30 + "\n")

    # --- AFTER Optimization ---
    print("--- Running OPTIMIZED Program (After Compilation) ---")
    optimized_pred = optimized_program(transaction=test_transaction)
    print(f"Input: '{test_transaction}'")
    print(f"Output (optimized): {optimized_pred.account}")

    # Inspect the new prompt. You'll see it now includes the 2 few-shot examples
    # that the teleprompter generated!
    print("\n--- Prompt (optimized) ---")
    optimized_program.inspect_history(n=1)


if __name__ == "__main__":
    main()
