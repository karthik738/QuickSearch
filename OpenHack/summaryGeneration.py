# T5 MODEL
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


def generate_summary(full_content, max_token_limit):

    # Tokenize the input text
    inputs = tokenizer.encode(
        "summarize: " + full_content,
        return_tensors="pt",
        max_length=1024,
        truncation=False,
    )

    # Generate the summary
    summary_ids = model.generate(
        inputs,
        max_length=max_token_limit,
        min_length=100,
        length_penalty=2.0,
        num_beams=5,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# # Initialize the tokenizer and model
# tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
# model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')

# # Input text to summarize
# def generate_summary(full_content, max_token_limit):
#     # Tokenize the input text
#     inputs = tokenizer(full_content, return_tensors="pt", max_length=1024, truncation=True)
#     # Generate the summary
#     summary_ids = model.generate(inputs['input_ids'], max_length=max_token_limit, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary
