# OPEN AI (GPT 2, GPT 3, ROBERTA - TRANSFORMER BASED MOEDLS)
# import openai
# openai.api_key = 'YOUR_API_KEY'

# def generate_summary(full_content,max_token_limit):
#     prompt = f"Summarize the following article:\n{full_content}"

#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=prompt,
#         temperature=0.7,
#         max_tokens=max_token_limit
#     )

#     summary = response.choices[0].text.strip()
#     return summary

# T5 MODEL  (HUGGING FACE TRANSFORMER MODEL)
# from transformers import T5Tokenizer, T5ForConditionalGeneration

# # Initialize the tokenizer and model
# tokenizer = T5Tokenizer.from_pretrained('t5-small')
# model = T5ForConditionalGeneration.from_pretrained('t5-small')

# # Input text to summarize
# input_text = ""
# # Tokenize the input text
# inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)

# # Generate the summary
# summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
# summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# print(summary)


# PEGASUS GOOGLE
# from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# # Initialize the tokenizer and model
# tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
# model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')

# # Input text to summarize
# input_text = ""

# # Tokenize the input text
# inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

# # Generate the summary
# summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
# summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# print(summary)


# BART (FACEBOOK)
# from transformers import BartTokenizer, BartForConditionalGeneration

# # Initialize the tokenizer and model
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

# # Input text to summarize
# input_text = ""

# # Tokenize the input text
# inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

# # Generate the summary
# summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
# summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# print(summary)

# BERT SUM(PYTORCH)
