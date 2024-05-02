import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources (if not already downloaded)
nltk.download("punkt")

# Input text
input_text = "This is a sample input text."

# Tokenize the input text using NLTK
tokens = word_tokenize(input_text)

# Print the tokens
print("NLTK Tokens:", tokens)
