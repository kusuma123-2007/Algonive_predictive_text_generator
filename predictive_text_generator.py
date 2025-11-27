"""
TASK 1 – PREDICTIVE TEXT GENERATOR
----------------------------------

Key Features Implemented:

1. Word Prediction
   - Suggests likely next words based on what the user has typed.

2. Context Awareness
   - Uses previous words (bigram + trigram model) to improve accuracy.
   - Tries trigram (last 2 words), falls back to bigram (last word),
     then to most frequent words (unigram).

3. Customizable Dictionary
   - User can add their own sentences/phrases.
   - These are immediately learned by the model and influence predictions.
   - Also shows list of custom words.

4. Basic Machine Learning
   - Uses a simple n-gram Markov model:
       - Unigrams, Bigrams, Trigrams built from training corpus
       - Counts are used as probabilities for predictions.
"""

import re
from collections import defaultdict, Counter


class PredictiveTextGenerator:
    def __init__(self):
        # Unigram: word -> count
        self.unigram_counts = Counter()

        # Bigram: current_word -> Counter(next_word -> count)
        self.bigram_counts = defaultdict(Counter)

        # Trigram: (prev_word, current_word) -> Counter(next_word -> count)
        self.trigram_counts = defaultdict(Counter)

        # Words added by the user (for customizable dictionary feature)
        self.custom_words = set()

    # -----------------------------
    # TEXT PREPROCESSING
    # -----------------------------
    @staticmethod
    def preprocess(text: str):
        """
        Lowercase text, keep only letters and spaces, split into tokens.
        """
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = text.split()
        return tokens

    # -----------------------------
    # TRAINING / UPDATING MODEL
    # -----------------------------
    def train_on_text(self, text: str):
        """
        Build n-gram counts from the given training text.
        """
        tokens = self.preprocess(text)
        self._update_counts(tokens)

    def add_to_dictionary(self, user_text: str):
        """
        Add user-provided sentence/phrase to the model
        (Customizable Dictionary feature).
        """
        tokens = self.preprocess(user_text)

        # Track custom words for display
        for w in tokens:
            self.custom_words.add(w)

        # Update n-gram counts so the model "learns" from user text
        self._update_counts(tokens)

    def _update_counts(self, tokens):
        """
        Internal helper: update uni/bi/tri-gram counts given token list.
        """
        if not tokens:
            return

        # Unigram counts
        for w in tokens:
            self.unigram_counts[w] += 1

        # Bigram counts
        for i in range(len(tokens) - 1):
            w1 = tokens[i]
            w2 = tokens[i + 1]
            self.bigram_counts[w1][w2] += 1

        # Trigram counts
        for i in range(len(tokens) - 2):
            w1 = tokens[i]
            w2 = tokens[i + 1]
            w3 = tokens[i + 2]
            self.trigram_counts[(w1, w2)][w3] += 1

    # -----------------------------
    # PREDICTION LOGIC (MARKOV MODEL)
    # -----------------------------
    def predict_next_words(self, prefix: str, top_k: int = 5):
        """
        Predict top_k next words given the prefix using:
        1. Trigram context (last 2 words), else
        2. Bigram context (last word), else
        3. Most frequent words (unigram)
        """
        tokens = self.preprocess(prefix)

        # 1. Try trigram (context: last two words)
        if len(tokens) >= 2:
            key = (tokens[-2], tokens[-1])
            if key in self.trigram_counts:
                candidates = self.trigram_counts[key]
                return self._top_k_from_counter(candidates, top_k)

        # 2. Try bigram (context: last one word)
        if len(tokens) >= 1:
            key = tokens[-1]
            if key in self.bigram_counts:
                candidates = self.bigram_counts[key]
                return self._top_k_from_counter(candidates, top_k)

        # 3. Fallback: most frequent unigrams
        return self._top_k_from_counter(self.unigram_counts, top_k)

    @staticmethod
    def _top_k_from_counter(counter_obj, k):
        """
        Helper to get top-k items from a Counter or similar mapping.
        Returns only the words, sorted by decreasing count.
        """
        if not counter_obj:
            return []
        # counter_obj.most_common(k) returns list of (word, count)
        return [word for word, _ in counter_obj.most_common(k)]


# -----------------------------
# INITIAL TRAINING CORPUS
# -----------------------------
DEFAULT_TRAINING_TEXT = """
Predictive text generators are widely used in messaging applications.
When users type a word, the system predicts the next word.
Machine learning and natural language processing techniques
help to build accurate predictive text models.
These models learn from previous sentences, user behaviour,
and frequently used words. Autocomplete features in chat apps,
email clients, and search engines are examples of predictive text.
The more the system is used, the better its predictions become.
"""


# -----------------------------
# COMMAND-LINE INTERFACE (CLI)
# -----------------------------
def main():
    generator = PredictiveTextGenerator()

    # Train on default corpus
    generator.train_on_text(DEFAULT_TRAINING_TEXT)

    print("=" * 60)
    print(" PREDICTIVE TEXT GENERATOR  (Python + Markov N-gram Model)")
    print("=" * 60)
    print("Key Features:")
    print("  1) Word prediction")
    print("  2) Context awareness (uses previous words)")
    print("  3) Customizable dictionary (learns from your phrases)")
    print("  4) Basic machine learning using n-gram counts\n")

    while True:
        print("\nChoose an option:")
        print("  1) Get next-word predictions")
        print("  2) Add your own sentence/phrase (custom dictionary)")
        print("  3) Show custom words added")
        print("  4) Exit")
        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            # WORD PREDICTION + CONTEXT AWARENESS
            user_text = input("\nType your sentence (or partial sentence):\nYou: ")
            suggestions = generator.predict_next_words(user_text, top_k=5)

            if not suggestions:
                print("No suggestions available yet.")
            else:
                print("Suggestions:", ", ".join(suggestions))

        elif choice == "2":
            # CUSTOMIZABLE DICTIONARY
            custom_text = input(
                "\nEnter a sentence/phrase that you use frequently:\n> "
            )
            generator.add_to_dictionary(custom_text)
            print("Your sentence has been learned by the model!")

        elif choice == "3":
            # SHOW CUSTOM WORDS
            if not generator.custom_words:
                print("\nNo custom words added yet.")
            else:
                print("\nCustom words you have added:")
                for w in sorted(generator.custom_words):
                    print(" -", w)

        elif choice == "4":
            print("\nExiting. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
