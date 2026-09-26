import re
import string

LIKELIHOOD_SCAM_WORDS = {"bank", "fraud", "scam", "transfer", "prize", "winner"}

def word_count(string):
    return len(string.split())


def sentence_count(string):
    return len(re.split(r'[.!?]+', string)) - 1

def punctuation_count(string):
   return {
       "commas": string.count(","),
       "periods": string.count("."),
       "exclamations": string.count("!"),
       "questions": string.count("?")
   }

def check_scam_words(string):
    scam_words = {}
    for word in LIKELIHOOD_SCAM_WORDS:
        scam_words[word] = string.count(word)
    return scam_words

def char_length(string):
    return len(string)

def avg_word_length(string):
    words = string.split()
    return sum(len(w) for w in words) / len(words) if words else 0

def uppercase_ratio(string):
    return sum(1 for c in string if c.isupper()) / len(string) if string else 0

def digit_ratio(string):
    return sum(1 for c in string if c.isdigit()) / len(string) if string else 0

def url_count(string):
    return len(re.findall(r'http[s]?://\S+', string))

def email_count(string):
    return len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', string))

def vocabulary_diversity(string):
    words = string.lower().split()
    return len(set(words)) / len(words) if words else 0

def check_string(string):
    return {"word_count": word_count(string),
            "sentence_count": sentence_count(string),
            "punctuation_count": punctuation_count(string),
            "check_scam_words": check_scam_words(string),
            "avg_word_length": avg_word_length(string),
            "char_length": char_length(string),
            "uppercase_ratio": uppercase_ratio(string),
            "digit_ratio": digit_ratio(string),
            "url_count": url_count(string),
            "email_count": email_count(string),
            "vocabulary_diversity": vocabulary_diversity(string),
            }

def main():
    return -1

if __name__ == "__main__":
    main()