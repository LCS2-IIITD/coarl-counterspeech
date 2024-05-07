import os

TRAIN = os.path.join("./", "data/IntentConanV2/train.csv")
TEST = os.path.join("./", "data/IntentConanV2/test.csv")
DEV = os.path.join("./", "data/IntentConanV2/validation.csv")

ROOT = os.path.join("./")
OPENAI_CREDS = os.path.join("./", "creds/openai.json")

HATESPEECH_COL = "hatespeech"
COUNTERSPEECH_COL = "counterspeech"
INTENT_COL = "csType"
ID_COL = "id"
HATESPEECH_EXP_COLS = [
    "hatespeechOffensiveness",
    "targetGroup",
    "speakerIntent",
    "relevantPowerDynamics",
    "hatespeechImplication",
    "targetGroupEmotionalReaction",
    "targetGroupCognitiveReaction",
]

TEST_SIZE = 0.3
RANDOM_STATE = 1998
MODEL_NAME = "flant-t5-xxl"
MODEL_TYPE = "t5"
