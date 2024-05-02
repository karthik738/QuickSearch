import spacy

# Load the English tokenizer
nlp = spacy.load("en_core_web_sm")

# Input text
input_text = "This is a sample input text."

# Tokenize the input text using spaCy
doc = nlp(input_text)

# Extract tokens
tokens = [token.text for token in doc]

# Print the tokens
print("spaCy Tokens:", tokens)
