import json
import os
import random
import re
import string

import numpy as np
import pandas as pd
import torch
from cleantext import clean
from loguru import logger
from sklearn.model_selection import train_test_split

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str, min_len=4) -> str:
    """
    Returns cleaned text (post/title) by removing extra whitespaces, unprintable characters, recurring punctions, & urls

    Before:

    My ðŸ¤ [ndad] (https://np.reddit.com/r/raisedbynarcissists/comments/4puzqb/if_ndad_shows_up_to_my_graduation_im_going_to/) and I have been estranged for about 4-5 months now. Since the estrangement, I have noticed feeling this desire to tighten my abdominal muscles when I remember something my ndad did or didn't do that led me to feel so much pain as a result of feeling angry, hurt or depressed by all the hurt he has caused me.

    The ¤ only other time I think of tightening my abdominal muscles is during exercise because I remember when I used exercise videos, the instructor would instruct such...!

    I'm taking medications ([at the very least, I am trying](https://np.reddit.com/r/Anger/comments/4rjatz/new_doctor_wants_me_to_go_off_antidepressants/)) and am going to therapy so I hope I am handling this pretty well, just wanted to get second opinions.

    Should I be concerned about this.......? Do you have any idea why one would desire to tighten abdominal muscles in this situation or in situations such as these......!?
    Ÿ


    *********************************************************************************************************************************

    After:
    my [ndad] (<url>) and i have been estranged for about 4-5 months now. since the estrangement, i have noticed feeling this desire to tighten my abdominal muscles when i remember something my ndad did or didn't do that led me to feel so much pain as a result of feeling angry, hurt or depressed by all the hurt he has caused me. the  only other time i think of tightening my abdominal muscles is during exercise because i remember when i used exercise videos, the instructor would instruct such! i'm taking medications ([at the very least, i am trying](<url>)) and am going to therapy so i hope i am handling this pretty well, just wanted to get second opinions. should i be concerned about this? do you have any idea why one would desire to tighten abdominal muscles in this situation or in situations such as these?

    """

    punc = """()-[]{};:'"\<>/@#$%^&*_~"""

    if not isinstance(text, str):
        logger.debug(ValueError(f"encountered invalid format string: {text}"))
        return None

    text = re.sub(
        f"[^{re.escape(string.printable)}]",
        "",
        re.sub(
            r"[\?\.\!]+(?=[\?\.\!])",
            "",
            clean(
                text,
                fix_unicode=False,
                to_ascii=False,
                lower=False,
                no_line_breaks=True,
                no_urls=True,
                no_emails=True,
                no_punct=False,
                no_emoji=True,
                no_currency_symbols=True,
                lang="en",
                replace_with_url="",
                replace_with_email="",
            ),
        ),
    ).strip()

    for _ in punc:
        text = text.replace(_, "")
    text = text.replace(f". .", ". ").replace("  ", " ")
    text = text.strip()

    if len(text) < min_len:
        return "<EMPTY_TEXT>"
    else:
        return text.strip()


def load_json(fpath: str):
    if not fpath.endswith(".json"):
        raise ValueError(f"{fpath} not a json file")

    with open(fpath, "r") as fp:
        return json.load(fp)


def save_json(data, fpath: str):
    if not fpath.endswith(".json"):
        raise ValueError(f"{fpath} not a json file")

    with open(fpath, "w") as fp:
        return json.dump(data, fp)


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    id_col: str,
    stratify_col: str,
    random_state: int,
):
    x_train, x_test, y_train, y_test = train_test_split(
        df[id_col].values.tolist(),
        df[stratify_col].values.tolist(),
        stratify=df[stratify_col].values.tolist(),
        test_size=test_size,
        random_state=random_state,
    )

    df_train = df[df[id_col].isin(x_train)]
    df_test = df[df[id_col].isin(x_test)]
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()

    logger.debug(
        f"\nTrain set: {df_train.shape}\n{df_train[stratify_col].value_counts()}"
    )
    logger.debug(f"\nTest Set: {df_test.shape}\n{df_test[stratify_col].value_counts()}")

    return df_train, df_test


def search_best_match(search_text: str, df: pd.DataFrame, column: str):
    # Load the pre-trained sentence transformer model
    model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

    # Compute the embeddings for the search text and the DataFrame column
    search_text_embedding = model.encode([search_text])
    column_embeddings = model.encode(df[column].tolist())

    # Compute the cosine similarity between the embeddings
    similarities = cosine_similarity(search_text_embedding, column_embeddings)

    # Find the index of the best match
    best_match_index = similarities.argmax()

    # Return the best match
    return best_match_index, df.iloc[best_match_index]
