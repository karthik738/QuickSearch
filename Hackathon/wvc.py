from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import All_sheets
import re

reference_document = All_sheets.sheet_list[0]
print(reference_document)
reference_document = """Gypsum plaster must be started only after completion of water testing of above floors and completion of structure minimum three floors above.
All debris shall be removed and cleaned for centerline marking.
Before starting of plastering, all the wall tie holes, DSR holes, honeycombs and any extra
openings should be closed with approved non-shrink grout and cured enough.
Pond test must be completed for the ceiling / roof slab prior to the start of Gypsum plaster
In case of any leakages observed, waterproofing must be done and ensure watertight ceiling
Water jet test must be done from the outside for the wall tie hole packing and rectification must be done if there is any leakage. Ensure water tight walls prior to Gypsum plaster.
All ledge walls in rooms must be completed including C.M. plastering prior to start of Gypsum plaster
Bull Marking to be done for wall & ceiling at a time. If any chipping is required permission from PMC to be obtained as all the services has to be coordinated
Bull marks for minimum of one entire flat to be done before calling the PMC for checking.
Ceiling plaster and wall plaster to be done preferably by same mason and should happen simultaneously to avoid the undulations in the corners.
MEP clearance must be taken prior to plastering.
Electrical switch boxes must be filled with thermocol/silfil foam prior to plastering
Wall and ceiling surfaces should be properly hacked or approved bonding agent must be used
Top floor plastering to be done only after completion of terrace water proofing.
All wall conduits / pipes should be closed with non-shrink screed and fiber mesh and properly cured before start of plaster for walls and in coordination of services like electrical and plumbing.
Gypsum plaster/ putty should not be applied at dado portions of kitchen, wash basin & utility
Clean the surface from dirt, dust, grease, loose particles etc. and approved make of gypsum has to be applied for interior walls & ceiling and approved coarse putty must be applied for utility, sit out, corridor and staircase areas.
Electrical junction boxes must be kept open and should be closed prior to 1st coat painting
If plastering has gone up to 3rdfloor, ensure that plastering checklists must be closed in the lower floors.
Walls with undulations / plumb out above 12mm should be done with C.M plastering as a
leveling course and Gypsum plaster to be applied over and above C.M. plastering.
Termination of ends at openings should be done only after fixing of UPVC windows and door frames.
Mockup has to be done for every block and has to be approved.
Floor protection must be ensured by polythene sheet prior to start of gypsum plaster
Skirting top gypsum finishing should be done only after surface is dry to true level and line as per approved mock up.
Proper lighting to be provided during the work, if required.
Gypsum bags stacking in flats should be not more than 4 bags height.
Centre line / Reference line should be taken from the corridor and line perpendicular to the
center line has to be taken into the flats.
Centre line and offset line for each room shall be marked and Bull marking shall be done with reference to this offset line.
Sufficient bull marks should be made available & no where it should be less than 6mm as per manufacturer specification.
All the bulged portions should be chipped off in line with the gypsum bull marks.
Fiber mesh should be fixed rigidly at all the electrical & plumbing grooves over the screed
Ensure that meter level markings are made available on all walls.
Bonding agent should be mixed thoroughly prior to uniform application with approved roller over internal walls & ceilings.
Gypsum plaster should be completed within 3 days of bonding agent application or as per manufacturer’s recommendation.
Pre-wetting must be done for the cement mortar plaster surface (rectified surface) prior to gypsum plaster application.
Water should be added to the Gypsum powder as per the manufacturer specification and
prepared gypsum plaster mix should be consumed within the initial setting time as given by the manufacturer.
Undulations in plastered walls to be checked with aluminum straight edge and line-dory. Same
should be rectified immediately while the plaster is in fresh state.
Spilled plaster on Electrical junction boxes, on walls & floor should be cleared on the same day of work.
All door and window openings after plastering should be checked by templates as per specification.
Day to day plaster work waste/dead material shall be removed and proper housekeeping to be maintained at site.
Gypsum plastered surface should be air cured for 3 days.
Hollow sound test should be conducted on plastered surfaces to ensure proper bonding between walls and plaster."""


# Split the document into lines
lines = reference_document.split("\n")

# Convert each line to a sentence ending with a period
text_data = [re.sub(r"([^.])$", r"\1.", line.strip()) for line in lines]

# Preprocess the text data
processed_data = [preprocess(sentence) for sentence in text_data]

# Train the Word2Vec model
model = Word2Vec(processed_data, vector_size=200, window=5, min_count=1, sg=0)

# Save the trained model
model.save("word2vec.model")

# ----------------------------------------------------------------------------------


# Preprocess text
def preprocess(text):
    return [token for token in simple_preprocess(text.lower())]


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


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# def calculate_similarity_for_each_point(reference_document, human_document, threshold=0.5):
#     reference_points = reference_document.split("\n")
#     human_points = human_document.split("\n")

#     # Create TF-IDF vectors for reference and human documents
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform(reference_points + human_points)

#     # Calculate cosine similarity between each pair of points
#     similarity_matrix = cosine_similarity(tfidf_matrix[:len(reference_points)], tfidf_matrix[len(reference_points):])

#     scores = {}
#     for i, ref_point in enumerate(reference_points):
#         max_similarity = max(similarity_matrix[i])
#         if max_similarity > threshold:
#             human_point_index = similarity_matrix[i].argmax()
#             human_point = human_points[human_point_index]
#             scores[ref_point] = (max_similarity, human_point)
#         else:
#             scores[ref_point] = (0, None)

#     return scores


reference_document = """Gypsum plaster must be started only after completion of water testing of above floors and completion of structure minimum three floors above.
All debris shall be removed and cleaned for centerline marking.
Before starting of plastering, all the wall tie holes, DSR holes, honeycombs and any extra
openings should be closed with approved non-shrink grout and cured enough.
Pond test must be completed for the ceiling / roof slab prior to the start of Gypsum plaster
In case of any leakages observed, waterproofing must be done and ensure watertight ceiling
Water jet test must be done from the outside for the wall tie hole packing and rectification must be done if there is any leakage. Ensure water tight walls prior to Gypsum plaster.
All ledge walls in rooms must be completed including C.M. plastering prior to start of Gypsum plaster
Bull Marking to be done for wall & ceiling at a time. If any chipping is required permission from PMC to be obtained as all the services has to be coordinated
Bull marks for minimum of one entire flat to be done before calling the PMC for checking.
Ceiling plaster and wall plaster to be done preferably by same mason and should happen simultaneously to avoid the undulations in the corners.
MEP clearance must be taken prior to plastering.
Electrical switch boxes must be filled with thermocol/silfil foam prior to plastering
Wall and ceiling surfaces should be properly hacked or approved bonding agent must be used
Top floor plastering to be done only after completion of terrace water proofing.
All wall conduits / pipes should be closed with non-shrink screed and fiber mesh and properly cured before start of plaster for walls and in coordination of services like electrical and plumbing.
Gypsum plaster/ putty should not be applied at dado portions of kitchen, wash basin & utility
Clean the surface from dirt, dust, grease, loose particles etc. and approved make of gypsum has to be applied for interior walls & ceiling and approved coarse putty must be applied for utility, sit out, corridor and staircase areas.
Electrical junction boxes must be kept open and should be closed prior to 1st coat painting
If plastering has gone up to 3rdfloor, ensure that plastering checklists must be closed in the lower floors.
Walls with undulations / plumb out above 12mm should be done with C.M plastering as a
leveling course and Gypsum plaster to be applied over and above C.M. plastering.
Termination of ends at openings should be done only after fixing of UPVC windows and door frames.
Mockup has to be done for every block and has to be approved.
Floor protection must be ensured by polythene sheet prior to start of gypsum plaster
Skirting top gypsum finishing should be done only after surface is dry to true level and line as per approved mock up.
Proper lighting to be provided during the work, if required.
Gypsum bags stacking in flats should be not more than 4 bags height.
Centre line / Reference line should be taken from the corridor and line perpendicular to the
center line has to be taken into the flats.
Centre line and offset line for each room shall be marked and Bull marking shall be done with reference to this offset line.
Sufficient bull marks should be made available & no where it should be less than 6mm as per manufacturer specification.
All the bulged portions should be chipped off in line with the gypsum bull marks.
Fiber mesh should be fixed rigidly at all the electrical & plumbing grooves over the screed
Ensure that meter level markings are made available on all walls.
Bonding agent should be mixed thoroughly prior to uniform application with approved roller over internal walls & ceilings.
Gypsum plaster should be completed within 3 days of bonding agent application or as per manufacturer’s recommendation.
Pre-wetting must be done for the cement mortar plaster surface (rectified surface) prior to gypsum plaster application.
Water should be added to the Gypsum powder as per the manufacturer specification and
prepared gypsum plaster mix should be consumed within the initial setting time as given by the manufacturer.
Undulations in plastered walls to be checked with aluminum straight edge and line-dory. Same
should be rectified immediately while the plaster is in fresh state.
Spilled plaster on Electrical junction boxes, on walls & floor should be cleared on the same day of work.
All door and window openings after plastering should be checked by templates as per specification.
Day to day plaster work waste/dead material shall be removed and proper housekeeping to be maintained at site.
Gypsum plastered surface should be air cured for 3 days.
Hollow sound test should be conducted on plastered surfaces to ensure proper bonding between walls and plaster."""

human_document = """Before start gypsum plaster completed 3 floors structure work
complted total wall tie and tie rod hole packing
Honey Combs and any structure issues packing completed
Pond test completed before start the work
Any water leakages observed inform to water proofing team and rectify the water proofing team
jet test completed before start the work and start the work
And any structure issues like wall bend and celling undulation observed rectify before start work
Before start the work completed total structure issues like rough plaster for walls and ceiling
First need survey points and the survey points based to Right angle checking in total flat.
And maximum 12 minimum 8 for walls and maximum 10 and minimum 6 for celling based to bull mark completed.
bondit apply for before 1 day
And any eletrical joints mustly used fiber mesh
block work joints and any two different material joints apply the fiber mesh
And finally client checking completed start the work """

scores, total_scores, length = calculate_similarity_for_each_point(
    reference_document, human_document
)

# Calculate the similarity scores with a threshold of 0.5
# scores = calculate_similarity_for_each_point(reference_document, human_document, threshold=0.5)

# # Calculate the overall score
# overall_score = sum(score for score, _ in scores.values()) / len(scores)
# print(f"Overall Score: {overall_score}")


final_total_scores = 0
for point, (similarity, human_point) in scores.items():
    print(f"Reference Point: {point}")
    if similarity > 0.5:  # Adjust threshold as needed
        print(f"Human Document Point: {human_point}")
        print(f"Similarity Score: {similarity}")
        final_total_scores += similarity
    else:
        print("No matching point found in human document")
        # final_total_scores+=0
    print()

print(total_scores * 10)
print((final_total_scores / length) * 100)


# def calculate_overall_score(reference_document, human_document):
#     reference_points = reference_document.split("\n")
#     human_points = human_document.split("\n")

#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform(reference_points + human_points)
#     similarity_matrix = cosine_similarity(tfidf_matrix[:len(reference_points)], tfidf_matrix[len(reference_points):])

#     total_score = 0
#     matched_points = 0
#     for i, ref_point in enumerate(reference_points):
#         max_similarity = max(similarity_matrix[i])
#         if max_similarity > 0.5:  # Threshold for similarity
#             total_score += max_similarity
#             matched_points += 1

#     if matched_points == 0:
#         return 0.0

#     return total_score / matched_points

# # Calculate the overall score for the human document
# overall_score = calculate_overall_score(reference_document, human_document)
# print(f"Overall Score: {overall_score*100}")
