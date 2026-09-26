import pandas as pd
import json
import pathlib
import ollama

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Load test set
test_df = pd.read_csv(DATA_DIR / "label_test.csv")

# Get fraud emails only
fraud_emails = test_df[test_df['label'] == 1]['text_combined'].head(100)

training_data = []

for i, email in enumerate(fraud_emails):
    response = ollama.generate(
        model="llama3.2:3b",
        prompt=f"""Analyze this email and provide a brief 1-2 sentence explanation of why it's fraudulent. 
Focus on specific fraud indicators (urgency, suspicious requests, impersonation, etc.).

Email: {email}

Explanation:"""
    )

    explanation = response["response"].strip()

    training_data.append({
        "email": email,
        "explanation": explanation
    })

    if (i + 1) % 10 == 0:
        print(f"Generated {i + 1}/{len(fraud_emails)} explanations")

# Save
output_path = DATA_DIR / "fraud_explanations.json"
with open(output_path, 'w') as f:
    json.dump(training_data, f, indent=2)

print(f"Saved {len(training_data)} email-explanation pairs")