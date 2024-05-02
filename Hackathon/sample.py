import All_sheets
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

ultimate_score=0
ultimate_manual_score=0
ultimate_score_threshold=0
ulimate_mae=0
for i in range(len(All_sheets.sheet_list)):
    reference_document = All_sheets.sheet_list[i].r_text
    human_document = All_sheets.sheet_list[i].e_text
    ultimate_manual_score+=All_sheets.sheet_list[i].ms

    # Preprocess text
    def preprocess(text):
        return [token for token in simple_preprocess(text.lower())]


    # Split the document into lines
    lines = reference_document.split("\n")

    # Convert each line to a sentence ending with a period
    text_data = [re.sub(r"([^.])$", r"\1.", line.strip()) for line in lines]

    # Preprocess the text data
    processed_data = [preprocess(sentence) for sentence in text_data]

    # Train the Word2Vec model
    model = Word2Vec(processed_data, vector_size=100, window=5, min_count=1, sg=0)

    # Save the trained model
    model.save("word2vec.model")

    # Load Word2Vec model
    model = Word2Vec.load("word2vec.model")


    def calculate_similarity_for_each_point(reference_document, human_document):
        reference_points = reference_document.split("\n")
        human_points = human_document.split("\n")

        scores = {}
        total_score = 0
        for ref_point in reference_points:
            ref_point_processed = preprocess(ref_point)
            max_similarity = 0
            matching_point = ""
            for human_point in human_points:
                human_point_processed = preprocess(human_point)
                similarity = model.wv.n_similarity(
                    ref_point_processed, human_point_processed
                )
                if similarity > max_similarity:
                    max_similarity = similarity
                    matching_point = human_point
            scores[ref_point] = (max_similarity, matching_point)
            total_score += max_similarity * 10
        # print(len(reference_points))
        total_score = total_score / len(reference_points)

        return scores, total_score, len(reference_points)
    
    scores, total_scores, length = calculate_similarity_for_each_point(
    reference_document, human_document)

    final_total_scores = 0
    for point, (similarity, human_point) in scores.items():
        # print(f"Reference Point: {point}")
        if similarity > 0.5:  # Adjust threshold as needed
            # print(f"Human Document Point: {human_point}")
            # print(f"Similarity Score: {similarity}")
            final_total_scores += similarity
        else:
            # print("No matching point found in human document")
            final_total_scores+=0
        # print()

    # print(total_scores * 10)
    # print((final_total_scores / length) * 100)
    ultimate_score+=total_scores*10
    ultimate_score_threshold+=(final_total_scores/length)*100
    print(f"ManualScore {i}:",All_sheets.sheet_list[i].ms)
    print(f"CalScore {i}:",(final_total_scores/length)*100)
    print(f"Score {i}:",total_scores*10)
    ulimate_mae+=abs(All_sheets.sheet_list[i].ms-(final_total_scores/length)*100)

# print(ultimate_manual_score/len(All_sheets.sheet_list))
# # print(ultimate_score/len(All_sheets.sheet_list))
# print(ultimate_score_threshold/len(All_sheets.sheet_list))
# print(ulimate_mae/len(All_sheets.sheet_list))


