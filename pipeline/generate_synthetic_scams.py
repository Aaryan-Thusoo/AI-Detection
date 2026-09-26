import ollama
import pandas as pd

SCAM_SCENARIOS = {
    "romance_scam": "Write a message from someone pretending to have a romantic online relationship, "
                    "who is now asking their partner for money due to a fabricated emergency.",
    "fake_invoice": "Write a fake invoice or billing email pretending to be from a real company, "
                    "pressuring the recipient to pay an overdue amount immediately.",
    "business_email_compromise": "Write an email impersonating a company executive, urgently asking an "
                                 "employee to process a wire transfer or purchase gift cards.",
    "phishing_link": "Write a phishing email pretending to be from a bank or service provider, asking "
                     "the recipient to click a link and verify their account details.",
    "lottery_inheritance": "Write a message claiming the recipient has won a lottery or inherited money "
                           "from a distant relative, asking for personal/banking details to release funds.",
    "tech_support_scam": "Write a message pretending to be tech support, claiming the recipient's computer "
                         "is infected and they need to pay for a fix or provide remote access.",
    "crypto_investment": "Write a message promoting a fake cryptocurrency investment opportunity promising "
                         "guaranteed high returns.",
}

MODEL = "llama3.2:3b"
MESSAGES_PER_SCENARIO = 20


def generate_messages():
    rows = []
    for scam_type, prompt in SCAM_SCENARIOS.items():
        successful = 0
        attempts = 0
        max_attempts = MESSAGES_PER_SCENARIO * 2

        while successful < MESSAGES_PER_SCENARIO and attempts < max_attempts:
            try:
                response = ollama.generate(
                    model=MODEL,
                    prompt=f"{prompt}",
                    options={"temperature": 0.9},
                )
                text = response["response"].strip()
                if text and len(text) > 20 and not any(x in text.lower() for x in ["can't", "cannot", "i can"]):
                    rows.append({
                        "text_combined": text,
                        "label": 1,
                        "ai_generated": 1,
                        "scam_type": scam_type,
                    })
                    successful += 1
            except Exception as e:
                pass

            attempts += 1

        print(f"{scam_type}: {successful}/{MESSAGES_PER_SCENARIO} generated")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_messages()
    df.to_csv("data/raw/ai_generated_scams.csv", index=False)
    print(f"Saved {len(df)} synthetic scam messages")