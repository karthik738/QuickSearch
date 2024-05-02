import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

reference_document = [
    "point 1: Hi, I am karthik. I am studying BTech at Kmit. I am in my 3rd year. ",
    "point 2: Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "point 3: Nulla facilisi. Vestibulum ac diam sit amet quam vehicula elementum sed sit amet dui.",
    "point 4: Etiam eget ligula eu lectus lobortis condimentum.",
    "point 5: Donec sollicitudin molestie malesuada.",
]

human_document = [
    "Elaboration of point 1: Karthik is studying Btech at Kmit",
    "Elaboration of point 1: Karthik is in his 3rd year",
    "Elaboration of point 2: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Elaboration of point 3: Nulla facilisi. Vestibulum ac diam sit amet quam vehicula elementum sed sit amet dui. Donec sollicitudin molestie malesuada.",
    "Elaboration of point 4: Etiam eget ligula eu lectus lobortis condimentum. Cras ultricies ligula sed magna dictum porta. Nulla porttitor accumsan tincidunt.",
    "Elaboration of point 5: Donec sollicitudin molestie malesuada. Curabitur non nulla sit amet nisl tempus convallis quis ac lectus.",
]


def preprocess(text):
    return " ".join(
        [
            token.text.lower()
            for token in nlp(text)
            if not token.is_stop and not token.is_punct
        ]
    )


def calculate_similarity(reference_document, human_document):
    vectorizer = TfidfVectorizer(stop_words="english")
    corpus = reference_document + human_document
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(
        tfidf_matrix[: -len(human_document)], tfidf_matrix[-len(human_document) :]
    )
    return similarity_matrix


def map_points(similarity_matrix):
    mapping = {}
    for i, row in enumerate(similarity_matrix):
        reference_index = row.argmax()
        if reference_index in mapping:
            mapping[reference_index].append(i)
        else:
            mapping[reference_index] = [i]
    return mapping


# def score_mapping(mapping, similarity_matrix):
#     scores = {}
#     for ref_index, human_indices in mapping.items():
#         total_similarity = sum([similarity_matrix[human_index][ref_index] for human_index in human_indices])
#         average_similarity = total_similarity / len(human_indices)
#         scores[ref_index] = average_similarity
#     return scores

# def score_mapping(mapping, similarity_matrix, reference_document, human_document):
#     scores = {}
#     for ref_index, human_indices in mapping.items():
#         total_similarity = 0
#         for human_index in human_indices:
#             human_sentence = human_document[human_index].split(': ')[1]  # Extract the sentence text without the prefix
#             ref_sentence = reference_document[ref_index].split(': ')[1]  # Extract the sentence text without the prefix
#             total_similarity += similarity_matrix[human_index][ref_index]
#         average_similarity = total_similarity / len(human_indices)
#         scores[ref_index] = average_similarity
#     return scores


def score_mapping(mapping, similarity_matrix, reference_document, human_document):
    scores = {}
    for ref_index, human_indices in mapping.items():
        total_similarity = 0
        for human_index in human_indices:
            if ref_index < len(reference_document) and human_index < len(
                human_document
            ):
                human_sentence = human_document[human_index].split(": ")[
                    1
                ]  # Extract the sentence text without the prefix
                ref_sentence = reference_document[ref_index].split(": ")[
                    1
                ]  # Extract the sentence text without the prefix
                total_similarity += similarity_matrix[human_index][ref_index]
        if len(human_indices) > 0:
            average_similarity = total_similarity / len(human_indices)
            scores[ref_index] = average_similarity
    return scores


# Preprocess documents
reference_document_preprocessed = [
    preprocess(sentence) for sentence in reference_document
]
human_document_preprocessed = [preprocess(sentence) for sentence in human_document]

# Calculate similarity matrix
similarity_matrix = calculate_similarity(
    reference_document_preprocessed, human_document_preprocessed
)

# Map points
mapping = map_points(similarity_matrix)

# Score mapping
# scores = score_mapping(mapping, similarity_matrix)

# Score mapping
# scores = score_mapping(mapping, similarity_matrix, reference_document, human_document)

# Score mapping
scores = score_mapping(mapping, similarity_matrix, reference_document, human_document)

# Print results
for ref_index, score in scores.items():
    print(f"Reference Point: {reference_document[ref_index]}, Score: {score}")
    for human_index in mapping[ref_index]:
        print(f" - Human Elaboration: {human_document[human_index]}")
