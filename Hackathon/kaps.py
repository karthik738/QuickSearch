from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from sklearn.neighbors import KDTree
import All_sheets


ultimate_score=0
ultimate_manual_score=0
ultimate_score_threshold=0
ulimate_mae=0

for i in range(len(All_sheets.sheet_list)):
    reference_document = All_sheets.sheet_list[i].r_text
    human_document = All_sheets.sheet_list[i].e_text
    ultimate_manual_score+=All_sheets.sheet_list[i].ms


    # Load the Sentence Transformer model
    model = SentenceTransformer("all-mpnet-base-v2")
    print("Model loaded till now, code is not slow")
    print(time.time())


    def remove_redundancy(vectors, threshold=0.9):
        # Use KDTree for more efficient nearest neighbor search
        kdtree = KDTree(vectors)
        unique_vectors = []

        # Find neighbors for all vectors in one query
        neighbors = kdtree.query_radius(vectors, r=1 - threshold)

        for i, vector in enumerate(vectors):
            # If no neighbors are found, the vector is unique
            if not neighbors[i].any():
                unique_vectors.append(vector)

        return unique_vectors


    def compare(employee_string, reference_string, threshold=0.75):
        employee_points = employee_string.split("\n")
        reference_points = reference_string.split("\n")

        total_marks = 100
        mark_per_point = total_marks / len(reference_points)

        employee_embeddings = model.encode(employee_points)
        reference_embeddings = model.encode(reference_points)

        # Remove redundancy more efficiently
        unique_employee_embeddings = remove_redundancy(employee_embeddings, threshold)

        dic = {}

        # Calculate cosine similarity in bulk
        similarities = cosine_similarity(
            np.array(reference_embeddings), np.array(unique_employee_embeddings)
        )
        normalized_similarities = 0.5 + 0.5 * similarities

        # Find the maximum similarity and its corresponding index for each reference point
        for i, reference_embedding in enumerate(normalized_similarities):
            max_similarity = np.max(reference_embedding)
            max_index = np.argmax(reference_embedding)

            if max_similarity > threshold:
                dic[i] = (max_similarity, max_index)

        achieved_points = mark_per_point * len(dic)

        return round(achieved_points, 2)
    
    result = compare(human_document,reference_document, threshold=0.7)
    # print(result)
    print(f"ManualScore {i}:",All_sheets.sheet_list[i].ms)
    print(f"CalScore {i}:",result)

