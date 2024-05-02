import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")


def calculate_similarity_for_each_point(reference_document, human_document):
    reference_points = reference_document.split("\n")
    human_points = human_document.split("\n")

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(reference_points + human_points)
    similarity_matrix = cosine_similarity(
        tfidf_matrix[: len(reference_points)], tfidf_matrix[len(reference_points) :]
    )

    scores = {}
    for i, ref_point in enumerate(reference_points):
        max_similarity = max(similarity_matrix[i])
        if max_similarity > 0.2:  # Threshold for similarity
            human_index = similarity_matrix[i].argmax()
            scores[ref_point] = (max_similarity, human_points[human_index])
        else:
            scores[ref_point] = (0, "")

    return scores


reference_document = """Before starting the flooring work , waterproofing to be completed in above three floors.
All the external plastering works to be completed before starting the flooring.
First coat putty work to be completed before flooring.
Tiles dado must be completed before flooring
FFL 1m level marking to be provided on all wall corners
All plumbing & electrical works to be completed including testing before flooring
MEP clerance must be take before flooring work
cleaning of the surface to be done before flooring
Tiles to be segregated batch wise to avoid shade variation
mockup of tiles workers has to be provided for every block on every week
Proper lighting to be provided if required.
GFC drawings to be pasted on the walls to ensure starting tile & pattern of the tile
Surface should be properly cleaned & pre wetted.
Cement mortar bed of 1:6(1cement:3MSand+3CRF) sand to be prepared
Before placing the tile cement slurry should be applied 2kg /sft.
Surface should be racked & tile should be placed.
After placing tile should be Tapped by wooden mallet to ensure air tight voids.
Spacers 3mm should be provided between each tile
6MM gap should be left from every wall.
8MM backer rod should be provided in the 6mm gap which we have left from each wall face to avoid damages due to temperature changes
Expansion joint should be provided for every 6m in corridor areas
Flat flooring level should be 10 more than corridor flooring level
In flats ,first we need to start bedroom flooring & then living & kitchen.
Curing should be done for floor tiles for 3 days
Hollowness should be checked after curing period & rectify them immediately.
Skirting tiles should be perpendicular to floor tiles.
Skew in the flooring to be rectified if any"""

human_document = """Tiling work shall be started only after completion of minimum three floors of waterproofing in each block.
Tiling work shall be started only after completion of external plastering of that block.
Toilet/Utility dado must be completed prior to starting of flooring.
Tiling work shall be started only after completion of first coat of putty
Before starting of tiling work, all the plumbing & electrical works should be completed including testing.
1 m level (from FFL) marking should be done on all walls.
Floor level for the entire floor has to be established prior to the starting of tiling activity.
Location of C.P & sanitary fittings, floor traps has to be ensured prior to starting of tiling work.
Block in charge should ensure rectification of waterproofing damage if any by the waterproofing agency.
Cleanliness of the area has to be ensured by the execution team prior to the commencement of tiling works.
Material to be checked and segregated batch wise prior to the commencement of work.
Ceramic tiles must be soaked for 2 hours prior to the usage.
Tiling Mockup has to be done for every block and has to be approved.
Proper lighting to be provided during the work, if required
FFL of the flat must be 10 mm more than FFL of the Corridor.
Lift sill level must be 10 mm more than corridor flooring and gentle slope has to be given from lift sill to corridor floor.
GFC drawings are to be pasted to the walls by clearly mentioning the details like where to start, pattern of tiling, slopes etc.
Door location has to be checked w.r.t. E.W.C. position of that toilet.
Mortar mix with 50% M-Sand and 50% CRF (CM 1:6 or as specified) has to be laid as bedding at the base of the tiles.
Tiles must be laid first in the bed rooms and then in the living/dining area (tiles should be laid from the SLD to main door, no trespassing should be allowed).
CC Bed to be laid below the SLD jamb portion to prevent damage of tile while fixing of UPVC SLD frame.
6mm gap to be maintained in tiling in all rooms/corridors along the walls to take care of thermal expansion/contraction.
Cement slurry @ 2 kg/ Sq. m has to be laid on the mortar uniformly.
Raking on the mortar has to be done so as to ensure proper bonding of the tile.
6mm expansion joints must be provided in the corridors as per the drawing and approved PU sealant should be used for these joints.
While laying of tiles, proper tamping with wooden /rubber mallet has to be ensured to remove the entrapped air.
Uniform cutting of tiles has to be done by mechanical means only, at floor traps.
During tiling care should be taken regarding the cutting of PVC pipes. Protection caps or end caps are to be provided to the UPVC pipes.
3 mm spacers are to be provided for both ceramic and vitrified flooring.
8 mm Backer rod should be provided into the 6mm gap along the walls at all rooms/corridors to take care of thermal expansion/contraction.
Skirting tiles should be chamfered by 3 to 5mm or as specified.
Skirting should be laid at right angle to the floor and uniform top line should be maintained.
Skirting adjacent to the architrave must be laid only after fixing of architrave.
Curing has to be done for 3 days and checking for slopes has to be done.
Hollowness check to be done, and if tiles with any hollowness or damages found, they are to be replaced with new ones or get them rectified.
Proper protection of tiles has to be ensured after completion of work.
Grouting to be done before handing over of the flat.
Epoxy grouting has to be done for all areas.
Checklist format to be filled prior, during and the post completion of tiling work."""

scores = calculate_similarity_for_each_point(reference_document, human_document)

for point, (similarity, human_point) in scores.items():
    print(f"Reference Point: {point}")
    if similarity > 0:
        print(f"Human Document Point: {human_point}")
        print(f"Similarity Score: {similarity}")
    else:
        print("No matching point found in human document")
    print()
