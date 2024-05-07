import gc
from collections import Counter
import pandas as pd

import evaluate
import numpy as np
import scipy
import torch
import torch.nn as nn
import ast
from bert_score import score as bert_score
from detoxify import Detoxify
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.util import ngrams
from pyemd import emd
from rouge import Rouge
from scipy.stats import gmean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    jaccard_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_metric

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
from category_accuracy import CI_Pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def free_memory():
    torch.cuda.empty_cache()
    gc.collect()


from utils import clean_text

from debater_python_api.api.debater_api import DebaterApi
import os


class Metrics:
    def __init__(self):
        self.label_list = ["Informative", "Questioning", "Positive", "Denouncing"]

    def compute_kendall_tau(self, predicted_rankings, true_rankings):
        # Kendall's Tau Rank Correlation Coefficient
        kendall_tau_values = []
        for pred_rank, true_rank in zip(predicted_rankings, true_rankings):
            pred_rank = [pred_rank[label] for label in self.label_list]
            true_rank = [true_rank[label] for label in self.label_list]

            kendall_tau = scipy.stats.kendalltau(pred_rank, true_rank).correlation
            kendall_tau_values.append(kendall_tau)

        average_kendall_tau = np.mean(kendall_tau_values)
        return average_kendall_tau

    def compute_spearman_rank(self, predicted_rankings, true_rankings):
        # Spearman's Rank Correlation Coefficient
        spearman_values = []
        for pred_rank, true_rank in zip(predicted_rankings, true_rankings):
            pred_rank = [pred_rank[label] for label in self.label_list]
            true_rank = [true_rank[label] for label in self.label_list]

            spearman_rho = scipy.stats.spearmanr(pred_rank, true_rank).correlation
            spearman_values.append(spearman_rho)

        average_spearman = np.mean(spearman_values)
        return average_spearman

    def compute_ndcg_score(self, predicted_rankings, true_rankings):
        def ndcg_at_k(r, k):
            """Compute NDCG for the top k rankings r.
            r is a list of binary relevances (1 for relevant, 0 for not relevant).
            """
            dcg_max = sum((1.0 / np.log(i + 2)) for i in range(k))
            dcg = sum((rel / np.log(i + 2)) for i, rel in enumerate(r[:k]))
            return dcg / dcg_max

        # Normalized Discounted Cumulative Gain (NDCG)
        ndcg_values = []
        for pred_rank, true_rank in zip(predicted_rankings, true_rankings):
            pred_rank = [pred_rank[label] for label in self.label_list]
            true_rank = [true_rank[label] for label in self.label_list]
            # Convert orders to binary relevance for some metrics
            predicted_rel = [1 if p == t else 0 for p, t in zip(pred_rank, true_rank)]

            ndcg = ndcg_at_k(predicted_rel, 4)
            ndcg_values.append(ndcg)

        average_ndcg = np.mean(ndcg_values)
        return average_ndcg

    def compute_precision_score(self, predicted_rankings, true_rankings):
        def precision_at_k(r, k):
            """Compute Precision at k.
            r is a list of binary relevances (1 for relevant, 0 for not relevant).
            """
            assert k >= 1
            return sum(r[:k]) / k

        def average_precision(r):
            """Compute the Average Precision.
            r is a list of binary relevances (1 for relevant, 0 for not relevant).
            """
            r = np.asarray(r)
            out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
            if not out:
                return 0.0
            return np.mean(out)

        # Normalized Discounted Cumulative Gain (NDCG)
        average_prec_scores = []
        precision_at_k_scores = []

        for pred_rank, true_rank in zip(predicted_rankings, true_rankings):
            pred_rank = [pred_rank[label] for label in self.label_list]
            true_rank = [true_rank[label] for label in self.label_list]
            # Convert orders to binary relevance for some metrics
            predicted_rel = [1 if p == t else 0 for p, t in zip(pred_rank, true_rank)]

            patk = precision_at_k(predicted_rel, 4)
            avg_prec = average_precision(predicted_rel)

            precision_at_k_scores.append(patk)
            average_prec_scores.append(avg_prec)

        average_prec_scores = np.mean(average_prec_scores)
        precision_at_k_scores = np.mean(precision_at_k_scores)

        return average_prec_scores, precision_at_k_scores

    def compute_f1_score(self, predicted_rankings, true_rankings):
        y_pred = [list(_)[0] for _ in predicted_rankings]
        y_true = [list(_)[0] for _ in true_rankings]
        f1 = f1_score(y_true, y_pred, average="weighted")
        return f1

    def compute_bleu(self, predictions, references):
        """
        compute the BLEU score for a prediction sentence given a reference sentence.
        :param reference: list of reference sentences
        :param prediction: prediction sentence
        :return: BLEU score
        """
        smoothie = SmoothingFunction().method4
        scores = []
        for prediction, reference in zip(predictions, references):
            score = sentence_bleu(reference, prediction, smoothing_function=smoothie)
            scores.append(score)
        return np.mean(scores)

    def compute_bleu_1(self, predictions, references):
        """
        compute the BLEU-1 score for a prediction sentence given a reference sentence.
        :param reference: list of reference sentences
        :param prediction: prediction sentence
        :return: BLEU-1 score
        """
        weights = (1.0, 0, 0, 0)
        smoothie = SmoothingFunction().method4
        scores = []
        for prediction, reference in zip(predictions, references):
            score = sentence_bleu(
                reference, prediction, weights=weights, smoothing_function=smoothie
            )
            scores.append(score)

        return np.mean(scores)

    def compute_bleu_2(self, predictions, references):
        """
        compute the BLEU-2 score for a prediction sentence given a reference sentence.
        :param reference: list of reference sentences
        :param prediction: prediction sentence
        :return: BLEU-2 score
        """
        weights = (0.5, 0.5, 0, 0)
        smoothie = SmoothingFunction().method4
        scores = []
        for prediction, reference in zip(predictions, references):
            score = sentence_bleu(
                reference, prediction, weights=weights, smoothing_function=smoothie
            )
            scores.append(score)

        return np.mean(scores)

    def compute_rouge_score(self, predictions, references):
        """
        compute the ROUGE score for a prediction sentence given a reference sentence.
        :param reference: reference sentence
        :param prediction: prediction sentence
        :return: ROUGE score
        """
        rouge = Rouge()
        rouge_l = []
        rouge_1 = []
        rouge_2 = []
        for prediction, reference in zip(predictions, references):
            rouge_l_score = rouge.get_scores(prediction, reference)[0]["rouge-l"]["f"]
            rouge_l.append(rouge_l_score)

            rouge_1_score = rouge.get_scores(prediction, reference)[0]["rouge-1"]["f"]
            rouge_1.append(rouge_1_score)

            rouge_2_score = rouge.get_scores(prediction, reference)[0]["rouge-2"]["f"]
            rouge_2.append(rouge_2_score)

        rouge_l = np.mean(rouge_l)
        rouge_1 = np.mean(rouge_1)
        rouge_2 = np.mean(rouge_2)

        return rouge_l, rouge_1, rouge_2

    def compute_bertscore(self, predictions, references, lang="en"):
        """
        Compute BERTScore for the given predictions and references.

        Args:
        - predictions (list[str]): List of predicted sentences.
        - references (list[str]): List of reference (ground truth) sentences.
        - model_type (str, optional): Type of BERT model to use. Default is "bert-base-multilingual-cased".
        - lang (str, optional): Language of the text. If not specified, it will be inferred from the model type.

        Returns:
        - P (torch.Tensor): Precision scores for each (prediction, reference) pair.
        - R (torch.Tensor): Recall scores for each (prediction, reference) pair.
        - F1 (torch.Tensor): F1 scores for each (prediction, reference) pair.
        """
        P, R, F1 = bert_score(predictions, references, lang=lang)
        free_memory()
        return F1.mean().item()

    def compute_self_bleu(self, predictions):
        """
        compute the Self-BLEU score for a list of generated sentences.
        :param candidates: list of generated sentences
        :return: Self-BLEU score
        """
        scores = []
        for prediction in predictions:
            candidates = prediction.split()
            references = [[c] for c in candidates]
            score = corpus_bleu(references, candidates)
            scores.append(score)

        return np.mean(scores)

    def compute_cosine_similarity(self, predictions, references):
        """
        compute the cosine similarity between two texts.
        :param reference: list of reference sentences
        :param prediction: list of prediction sentences
        :return: cosine similarity
        """
        vectorizer = TfidfVectorizer()
        scores = []
        for prediction, reference in zip(predictions, references):
            tfidf_matrix = vectorizer.fit_transform([reference, prediction])
            score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0][1]
            scores.append(score)

        return np.mean(scores)

    def compute_mauve(self, predictions, references):
        """
        compute the mauve score between two texts.
        :param reference: list of reference sentences
        :param prediction: list of prediction sentences
        :return: mauve score
        """
        mauve_scorer = evaluate.load("mauve")
        scores = []
        for prediction, reference in zip(predictions, references):

            score = mauve_scorer.compute(
                predictions=[prediction], references=[prediction]
            ).mauve
            scores.append(score)

        return np.mean(scores)

    def compute_bm25(self, predictions, references):
        """
        compute the BM25 similarity between two texts.
        :param reference: list of reference sentences
        :param prediction: list of prediction sentences
        :return: bm25 score
        """
        scores = []
        for prediction, reference in zip(predictions, references):
            # Tokenize the texts
            tokenized_text1 = word_tokenize(prediction.lower())
            tokenized_text2 = word_tokenize(reference.lower())

            # Create a corpus (in a real-world scenario, this should be a larger set of documents)
            corpus = [tokenized_text1, tokenized_text2]

            # Initialize BM25
            bm25 = BM25Okapi(corpus)

            # Querying with one text against the other
            query = tokenized_text1
            document_scores = bm25.get_scores(query)

            # Since we know the corpus has 2 documents, and we query with the first one,
            # the BM25 score for the second document is what we need
            score = document_scores[1]
            scores.append(score)

        return np.mean(scores)

    def compute_meteor_score_hf(self, predictions, references):
        """
        Compute the METEOR score for a prediction sentence given a reference sentence.
        :param reference: reference sentence
        :param prediction: prediction sentence
        :return: METEOR score
        """
        meteor = load_metric("meteor")
        scores = []
        for prediction, reference in zip(predictions, references):
            score = meteor.compute(predictions=[prediction], references=[reference])[
                "meteor"
            ]
            scores.append(score)

        average_meteor_score = np.mean(scores)
        return average_meteor_score

    def compute_novelty_score(self, predictions, references):
        def jaccard_similarity(text1, text2):
            set1 = set(text1.split())
            set2 = set(text2.split())
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            return len(intersection) / len(union)

        scores = []
        for prediction in tqdm(predictions, desc="Computing novelty score"):
            # Calculate the Jaccard similarity between the summary and the text
            jaccard_similarity_scores = [
                jaccard_similarity(prediction, reference) for reference in references
            ]
            # Return the novelty as 1 minus the maximum similarity
            novelty_score = 1 - max(jaccard_similarity_scores)
            scores.append(novelty_score)

        return np.mean(scores)

    def compute_diversity_score(self, predictions):
        def jaccard_similarity(text1, text2):
            set1 = set(text1.split())
            set2 = set(text2.split())
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            return len(intersection) / len(union)

        scores = []
        for prediction in tqdm(predictions, desc="Computing diversity score"):
            # Calculate the Jaccard similarity between the summary and the text
            jaccard_similarity_scores = [
                jaccard_similarity(prediction, x)
                for x in predictions
                if x != prediction
            ]
            # Return the novelty as 1 minus the maximum similarity
            diversity_score = 1 - max(jaccard_similarity_scores)
            scores.append(diversity_score)

        return np.mean(scores)

    def compute_inter_cn_repetition_score(self, predictions):
        scores = []
        for n in range(1, 9):
            ngram_counts = Counter()
            for text in predictions:
                tokens = text.split()
                ngram_counts.update(ngrams(tokens, n))
            total_ngrams = sum(ngram_counts.values())
            if total_ngrams == 0:
                scores.append(0)
            else:
                scores.append(
                    sum([count for ngram, count in ngram_counts.items() if count > 1])
                    / total_ngrams
                )

        return scores
        return gmean(scores)

    def compute_toxicity_score(self, predictions, batch_size=16):
        free_memory()
        model = Detoxify("unbiased", device="cuda")
        scores = {"toxicity": [], "obscene": [], "identity_attack": [], "insult": []}

        for i in tqdm(
            range(0, len(predictions), batch_size),
            desc=f"Running Toxicity inference on {len(predictions)} data points",
        ):
            batch_inputs = predictions[i : i + batch_size]
            outputs = model.predict(batch_inputs)
            scores["toxicity"].extend(outputs["toxicity"])
            scores["obscene"].extend(outputs["obscene"])
            scores["identity_attack"].extend(outputs["identity_attack"])
            scores["insult"].extend(outputs["insult"])

        return (
            np.mean(scores["toxicity"]),
            np.mean(scores["obscene"]),
            np.mean(scores["identity_attack"]),
            np.mean(scores["insult"]),
        )

    def compute_category_accuracy(
        self,
        predictions,
        groundtruth_categories,
        model_path="checkpoints/best_model-gpt-data-ca-fix2-roberta-large",
        batch_size=16,
    ):
        clf_pipeline = CI_Pipeline(model_path, use_cuda=True)
        y_pred = []

        for i in tqdm(
            range(0, len(predictions), batch_size),
            desc=f"Running CA inference on {len(predictions)} data points",
        ):
            batch_inputs = predictions[i : i + batch_size]
            outputs = clf_pipeline(batch_inputs, inference_type="cat_acc")
            y_pred.extend(outputs)

        acc = accuracy_score(y_pred=y_pred, y_true=groundtruth_categories)
        free_memory()
        return acc

    def compute_pc_cd_aq_scores(self, hatespeech, counterspeech, batch_size=100):
        key = "a6d8dfa763765e01663aaa2327891d6eL05"
        debater_api = DebaterApi(key)
        argument_quality_client = debater_api.get_argument_quality_client()
        pro_con_client = debater_api.get_pro_con_client()
        claim_detection_client = debater_api.get_claim_detection_client()

        pc_scores = []
        aq_scores = []
        cd_scores = []

        for i in tqdm(
            range(0, len(hatespeech), batch_size),
            desc=f"Running PC/CD/AQ inference on {len(hatespeech)} data points",
        ):
            batch_inputs_hatespeech = hatespeech[i : i + batch_size]
            batch_inputs_counterspeech = counterspeech[i : i + batch_size]

            input = [
                {"sentence": cs, "topic": hs}
                for cs, hs in zip(batch_inputs_hatespeech, batch_inputs_counterspeech)
            ]
            pc_scores.extend(pro_con_client.run(input))
            cd_scores.extend(claim_detection_client.run(input))
            aq_scores.extend(argument_quality_client.run(input))

        return pc_scores, cd_scores, aq_scores
        return np.mean(pc_scores), np.mean(cd_scores), np.mean(aq_scores)

    def get_classification_metrics(self, predicted_rankings, true_rankings):
        predicted_rankings = [
            ast.literal_eval(x) for x in predicted_rankings if isinstance(x, str)
        ]
        true_rankings = [
            ast.literal_eval(x) for x in true_rankings if isinstance(x, str)
        ]

        average_kendall_tau = self.compute_kendall_tau(
            predicted_rankings, true_rankings
        )

        average_spearman_rank = self.compute_spearman_rank(
            predicted_rankings, true_rankings
        )

        average_ndcg = self.compute_ndcg_score(predicted_rankings, true_rankings)
        average_precision, prec_at_k = self.compute_precision_score(
            predicted_rankings, true_rankings
        )
        f1_weighed = self.compute_f1_score(predicted_rankings, true_rankings)

        return {
            "average_kendall_tau": average_kendall_tau,
            "average_spearman_rank": average_spearman_rank,
            "average_precision": average_precision,
            "precision_at_4": prec_at_k,
            "average_ndcg": average_ndcg,
            "f1_weighed": f1_weighed,
        }

    def get_generation_metrics(self, df):
        print(f"Size of test set: {df.shape[0]}")
        if any(
            x for x in ["hatespeech", "csType", "counterspeech"] if x not in df.columns
        ):
            print(df.columns)
            raise Exception(
                f"Following columns are mandatory: { ['hatespeech', 'csType', 'counterspeech']}. Please make sure if these columns are missing"
            )

        print(f"Cleaning test data")

        hatespeech = (
            df["hatespeech"].apply(lambda x: clean_text(x).lower()).values.tolist()
        )
        counterspeech = (
            df["counterspeech"].apply(lambda x: clean_text(x).lower()).values.tolist()
        )
        predictions_m2 = (
            df["prediction_m2"].apply(lambda x: clean_text(x).lower()).values.tolist()
        )
        categories = df["csType"].values.tolist()

        # -------------------------------------------------------------------------------------------------------------------------------------------------#
        # """
        print("-" * 50)
        print(f"Calculating rouge score")
        rouge_l, rouge_1, rouge_2 = self.compute_rouge_score(
            predictions=predictions_m2, references=counterspeech
        )

        print("-" * 50)
        print(f"Calculating Self Bleu score")
        self_bleu_score_value = self.compute_self_bleu(predictions=predictions_m2)

        print("-" * 50)
        print(f"Calculating Repetition rate")
        inter_rr_score = self.compute_inter_cn_repetition_score(
            predictions=predictions_m2
        )

        print("-" * 50)
        print(f"Calculating Bleu1 and Bleu2 score")

        bleu_1_score_value = self.compute_bleu_1(
            predictions=predictions_m2, references=counterspeech
        )
        bleu_2_score_value = self.compute_bleu_2(
            predictions=predictions_m2, references=counterspeech
        )

        print("-" * 50)
        print(f"Calculating Cosine Similarity")
        cosine_similarity_value = self.compute_cosine_similarity(
            predictions=predictions_m2, references=counterspeech
        )

        print("-" * 50)
        print(f"Calculating Meteor score")
        meteor_score_value = self.compute_meteor_score_hf(
            predictions=predictions_m2, references=counterspeech
        )

        print("-" * 50)
        print(f"Calculating Bert score")
        bert_score_value = self.compute_bertscore(
            predictions=predictions_m2, references=counterspeech
        )

        print("-" * 50)
        print(f"Calculating Category Accuracy")
        cat_acc = self.compute_category_accuracy(
            predictions=predictions_m2, groundtruth_categories=categories
        )

        print("-" * 50)
        print(f"Calculating Toxicity score")
        (
            toxicity_score,
            obscene_score,
            identity_attack_score,
            insult_score,
        ) = self.compute_toxicity_score(predictions=predictions_m2)

        pc_score, cd_score, aq_score = self.compute_pc_cd_aq_scores(
            hatespeech=hatespeech, counterspeech=predictions_m2
        )
        pd_score = [-(pc) + cd for pc, cd in zip(pc_score, cd_score)]

        free_memory()

        return {
            "self_bleu": self_bleu_score_value,
            "repetition_rate": inter_rr_score,
            "obscenity": obscene_score,
            "identity_attack": identity_attack_score,
            "insult": insult_score,
            "bleu_1": bleu_1_score_value,
            "bleu_2": bleu_2_score_value,
            "cosine_similarity": cosine_similarity_value,
            "rouge_l": rouge_l,
            "rouge_1": rouge_1,
            "rouge_2": rouge_2,
            "meteor_score": meteor_score_value,
            "bert_score": bert_score_value,
            "category_accuracy": cat_acc,
            "toxicity": toxicity_score,
            "pc_score": pc_score,
            "cd_score": cd_score,
            "arg_quality_score": aq_score,
            "pd_score": pd_score,
        }
