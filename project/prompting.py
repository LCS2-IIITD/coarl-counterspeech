import argparse
import os
import pandas as pd

offensiveness_prompt = (
    lambda hatespeech: f"""
Analyze the offensiveness of the statement: {hatespeech}
""".strip()
)

target_group_prompt = (
    lambda hatespeech: f"""
Identify the group of people that the speaker is targeting or discriminating against in the offensive statement: {hatespeech}
""".strip()
)

speaker_intent_prompt = (
    lambda hatespeech: f"""
Analyze the speaker's intention behind writing the offensive statement: {hatespeech}
""".strip()
)

power_dynamics_prompt = (
    lambda hatespeech: f"""
Explain the underlying power dynamics between the speaker and the target group in the offensive statement: {hatespeech}
""".strip()
)

implication_prompt = (
    lambda hatespeech: f"""
Explain the implied meaning underlying the offensive statement: {hatespeech}
""".strip()
)

emotional_reaction_prompt = (
    lambda hatespeech: f"""
Describe how the target group might feel emotionally after reading or listening to the offensive statement: {hatespeech}
""".strip()
)

cognitive_reaction_prompt = (
    lambda hatespeech: f"""
Describe how the target group might react cognitively after reading or listening to the offensive statement: {hatespeech}
""".strip()
)

cs_generation_prompt = (
    lambda hatespeech, csType: f"""
Analyze the different aspects such as offensiveness, target group, stereotype, power dynamics, implied meaning, emotional, and cognitive reactions before writing a {csType} counterspeech for the offensive statement: {hatespeech}
""".strip()
)


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process input and output file paths.")

    # Add arguments for input and output file paths
    parser.add_argument("--input_file", required=True, help="Path to the input file")
    parser.add_argument("--output_file", required=True, help="Path to the output file")

    # Parse the command line arguments
    args = parser.parse_args()

    # Check if the input file exists
    if os.path.exists(args.input_file):
        print(f"Processing input file '{args.input_file}'")
        df = pd.read_csv(args.input_file)
        df["prompt_offensiveness"] = df.apply(
            lambda row: offensiveness_prompt(row["hatespeech"]), axis=1
        )
        df["prompt_target_group"] = df.apply(
            lambda row: target_group_prompt(row["hatespeech"]), axis=1
        )
        df["prompt_speaker_intent"] = df.apply(
            lambda row: speaker_intent_prompt(row["hatespeech"]), axis=1
        )
        df["prompt_power_dynamics"] = df.apply(
            lambda row: power_dynamics_prompt(row["hatespeech"]), axis=1
        )
        df["prompt_implication"] = df.apply(
            lambda row: implication_prompt(row["hatespeech"]), axis=1
        )
        df["prompt_emotional_reaction"] = df.apply(
            lambda row: emotional_reaction_prompt(row["hatespeech"]), axis=1
        )
        df["prompt_cognitive_reaction"] = df.apply(
            lambda row: cognitive_reaction_prompt(row["hatespeech"]), axis=1
        )
        df["prompt_cs_generation"] = df.apply(
            lambda row: cs_generation_prompt(row["hatespeech"], row["csType"]), axis=1
        )
        df.to_csv(args.output_file, index=False)

    else:
        print(f"Input file '{args.input_file}' does not exist.")


if __name__ == "__main__":
    main()


"""
Sample usage
python prompting.py 
"""
