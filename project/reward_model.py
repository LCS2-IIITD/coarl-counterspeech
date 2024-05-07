import pandas as pd
from tqdm import tqdm
from detoxify import Detoxify
from debater_python_api.api.debater_api import DebaterApi
import torch
import gc
import os

os.environ["http_proxy"] = "http://xen03.iitd.ac.in:3128"
os.environ["https_proxy"] = "http://xen03.iitd.ac.in:3128"


class RewardModel:
    def __init__(self):
        self.label_list = ["Informative", "Questioning", "Positive", "Denouncing"]
        self.toxicity_model = Detoxify("unbiased", device="cpu")
        self.project_debator_key = "a6d8dfa763765e01663aaa2327891d6eL05"

    def free_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def compute_pc_aq_scores(self, hatespeech, predicted_counterspeech, batch_size=100):
        debater_api = DebaterApi(self.project_debator_key)
        argument_quality_client = debater_api.get_argument_quality_client()
        pro_con_client = debater_api.get_pro_con_client()

        pc_scores = []
        aq_scores = []

        for i in tqdm(
            range(0, len(hatespeech), batch_size),
            desc=f"Running PC/CD/AQ inference on {len(hatespeech)} data points",
        ):
            batch_inputs_hatespeech = hatespeech[i : i + batch_size]
            batch_inputs_counterspeech = predicted_counterspeech[i : i + batch_size]

            input = [
                {"sentence": cs, "topic": hs}
                for cs, hs in zip(batch_inputs_hatespeech, batch_inputs_counterspeech)
            ]
            pc_scores.extend(pro_con_client.run(input))
            aq_scores.extend(argument_quality_client.run(input))

        return pc_scores, aq_scores

    def compute_toxicity_score(self, predicted_counterspeech, batch_size=16):
        scores = {"toxicity": [], "obscene": [], "identity_attack": [], "insult": []}

        for i in tqdm(
            range(0, len(predicted_counterspeech), batch_size),
            desc=f"Running Toxicity inference on {len(predicted_counterspeech)} data points",
        ):
            batch_inputs = predicted_counterspeech[i : i + batch_size]
            outputs = self.toxicity_model.predict(batch_inputs)
            scores["toxicity"].extend(outputs["toxicity"])

        return scores["toxicity"]

    def dynamic_reward_function(self, pc_score, aq_score, toxicity_score):
        return 1 / 3 * (((1 - pc_score) / 2) + aq_score + (1 - toxicity_score))

    def compute_rewards(self, hatespeech, predicted_counterspeech):
        assert len(hatespeech) == len(predicted_counterspeech)
        pc_scores, aq_scores = self.compute_pc_aq_scores(
            hatespeech=hatespeech, predicted_counterspeech=predicted_counterspeech
        )
        toxicity_scores = self.compute_toxicity_score(
            predicted_counterspeech=predicted_counterspeech
        )
        reward_scores = [
            self.dynamic_reward_function(pc_score, aq_score, toxicity_score)
            for pc_score, aq_score, toxicity_score in zip(
                pc_scores, aq_scores, toxicity_scores
            )
        ]
        return reward_scores, pc_scores, aq_scores, toxicity_scores
