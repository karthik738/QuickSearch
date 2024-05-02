from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# case_2: 0.81 | case_3: 0.78 | internal: 0.94 | correlation: 0.98 | mae: 3.68
model = SentenceTransformer("all-mpnet-base-v2")
print("Model loaded till now, code is not slow")

def remove_redundancy(vectors, threshold=0.94):
    unique_vectors = []
    for i, vector in enumerate(vectors):
        is_unique = True
        for j, other_vector in enumerate(vectors):
            if i != j:  # Skip comparing the vector with itself
                similarity = cosine_similarity(
                    vector.reshape(1, -1), other_vector.reshape(1, -1)
                )
                normalized_similarity = 0.5 + 0.5 * similarity[0][0]
                if normalized_similarity >= threshold:
                    is_unique = False
                    break
        if is_unique:
            unique_vectors.append(vector)
    return unique_vectors

def compare(employee_string, referece_string, case_2=0.81, case_3=0.78):
    employee_points = employee_string.split("\n")
    reference_points = referece_string.split("\n")

    total_marks = 100
    mark_per_point = total_marks / len(reference_points)

    employee_embeddings = model.encode(employee_points)
    employee_embeddings = remove_redundancy(employee_embeddings)
    reference_embeddings = model.encode(reference_points)

    dic_case2 = {}
    for i in range(len(reference_embeddings)):
        for j in range(len(employee_embeddings)):
            similarity = cosine_similarity(
                employee_embeddings[j].reshape(1, -1),
                reference_embeddings[i].reshape(1, -1),
            )
            normalized_similarity = 0.5 + 0.5 * similarity[0][0]

            if normalized_similarity > case_2:
                if i in dic_case2:
                    if normalized_similarity > dic_case2[i][0]:
                        dic_case2[i] = (normalized_similarity, j)
                else:
                    dic_case2[i] = (normalized_similarity, j)

    dic_case3 = {}
    for i in range(len(employee_embeddings)):
        for j in range(len(reference_embeddings)):
            similarity = cosine_similarity(
                employee_embeddings[i].reshape(1, -1),
                reference_embeddings[j].reshape(1, -1),
            )
            normalized_similarity = 0.5 + 0.5 * similarity[0][0]

            if normalized_similarity > case_3:
                if i in dic_case3:
                    if normalized_similarity > dic_case3[i][0]:
                        dic_case3[i] = (normalized_similarity, j)
                else:
                    dic_case3[i] = (normalized_similarity, j)

    all_case_2 = set(i for i in dic_case2)
    all_case_3 = set(dic_case3[i][1] for i in dic_case3)

    x = all_case_2.union(all_case_3)
    return len(x) * mark_per_point

def main(list_of_references, list_of_employees):
    scores = []

    for i in range(len(list_of_employees)):
        score = compare(list_of_employees[i], list_of_references[i])
        scores.append(score)
    
    return scores