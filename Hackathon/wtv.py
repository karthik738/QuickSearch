from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


# Preprocess text
def preprocess(text):
    return [token for token in simple_preprocess(text.lower())]


import re

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

# Join the sentences back into a single string
# processed_document = ' '.join(sentences)

# print(processed_document)

# Example text data
# text_data = [
#     "Before starting the flooring work, waterproofing to be completed in above three floors.",
#     "All the external plastering works to be completed before starting the flooring.",
#     "First coat putty work to be completed before flooring.",
#     "Tiles dado must be completed before flooring.",
#     "FFL 1m level marking to be provided on all wall corners.",
#     "All plumbing & electrical works to be completed including testing before flooring.",
#     "MEP clerance must be take before flooring work.",
#     "Cleaning of the surface to be done before flooring.",
#     "Tiles to be segregated batch wise to avoid shade variation.",
#     "Mockup of tiles workers has to be provided for every block on every week.",
#     "Proper lighting to be provided if required.",
#     "GFC drawings to be pasted on the walls to ensure starting tile & pattern of the tile.",
#     "Surface should be properly cleaned & pre wetted.",
#     "Cement mortar bed of 1:6(1cement:3MSand+3CRF) sand to be prepared.",
#     "Before placing the tile cement slurry should be applied 2kg /sft.",
#     "Surface should be racked & tile should be placed.",
#     "After placing tile should be Tapped by wooden mallet to ensure air tight voids.",
#     "Spacers 3mm should be provided between each tile.",
#     "6MM gap should be left from every wall.",
#     "8MM backer rod should be provided in the 6mm gap which we have left from each wall face to avoid damages due to temperature changes.",
#     "Expansion joint should be provided for every 6m in corridor areas.",
#     "Flat flooring level should be 10 more than corridor flooring level.",
#     "In flats, first we need to start bedroom flooring & then living & kitchen.",
#     "Curing should be done for floor tiles for 3 days.",
#     "Hollowness should be checked after curing period & rectify them immediately.",
#     "Skirting tiles should be perpendicular to floor tiles.",
#     "Skew in the flooring to be rectified if any."
# ]


# Preprocess the text data
processed_data = [preprocess(sentence) for sentence in text_data]

# Train the Word2Vec model
model = Word2Vec(processed_data, vector_size=200, window=5, min_count=1, sg=0)

# Save the trained model
model.save("word2vec.model")
