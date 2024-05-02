from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from sklearn.neighbors import KDTree

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
        if not neighbors[i]:
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


# Test the function with the updated threshold
e_text = """Before start gypsum plaster completed 3 floors structure work
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
r_text = """Gypsum plaster must be started only after completion of water testing of above floors and completion of structure minimum three floors above.
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
result = compare(e_text, r_text, threshold=0.7)
print(result)
