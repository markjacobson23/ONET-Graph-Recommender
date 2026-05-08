import pandas as pd
import json
from pathlib import Path
from src.utils.config import load_config, resolve_project_path

def load_featured_tables(input_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    occupation_nodes = pd.read_csv(input_dir / "featured_occupation_nodes.csv")
    skill_nodes = pd.read_csv(input_dir / "featured_skill_nodes.csv")
    occupation_skill_edges = pd.read_csv(input_dir / "occupation_skill_edges.csv")
    return occupation_nodes, skill_nodes, occupation_skill_edges

def load_candidate_profile(profile_path):
    with open(profile_path, "r") as file:
        data = json.load(file)
        return data["candidate_id"], data["candidate_name"], data["skills"]

def normalize_skill_name(name):
    return name.strip().lower()


def match_candidate_skills(candidate_skills, skill_nodes):

    skill_name_to_idx = {
        row["skill_name"].strip().lower(): row["skill_idx"]
        for _, row in skill_nodes.iterrows()
    }

    matched_skills = []
    unmatched_skills = []

    for skill_name in candidate_skills:
        skill = normalize_skill_name(skill_name)
        if skill in skill_name_to_idx:
            matched_skills.append(skill_name_to_idx[skill])
        else:
            unmatched_skills.append(skill_name)

    return matched_skills, unmatched_skills

def score_occupations(matched_skill_idxs, occupation_skill_edges):

    if not matched_skill_idxs:
        return pd.DataFrame(columns=["occupation_idx", "total_score"])

    matched_skill_edges = occupation_skill_edges[
        occupation_skill_edges["skill_idx"].isin(matched_skill_idxs)
    ].copy()

    matched_skill_edges["score"] = matched_skill_edges["importance"] * matched_skill_edges["level"]

    scored_occupations = (
        matched_skill_edges
        .groupby("occupation_idx")
        .agg(total_score=("score", "sum"))
        .reset_index()
    )

    occupation_scores = scored_occupations[["occupation_idx", "total_score"]]

    return occupation_scores

def rank_occupations(scores, occupation_nodes, top_k=10):
    ranked_occupations = (
        scores
        .merge(occupation_nodes, on="occupation_idx", how="left")
        .sort_values("total_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    occupation_ranks = ranked_occupations[["onetsoc_code","occupation_idx", "occupation_title", "total_score"]]

    return occupation_ranks

def main():

    # load config
    config = load_config()

    # resolve paths
    featured_tables_dir = resolve_project_path(
        config["paths"]["featured_tables_dir"]
    )
    candidate_profile_path = resolve_project_path(
        Path(config["paths"]["profiles_dir"])
        / config["profiles"]["default_candidate_profile"]
    )
    top_k = config["baseline"]["top_k"]


    # load featured tables
    occupation_nodes, skill_nodes, occupation_skill_edges = load_featured_tables(featured_tables_dir)

    candidate_id, candidate_name, candidate_skills = load_candidate_profile(candidate_profile_path)

    # match candidate skills to skill nodes
    matched_skills, unmatched_skills = match_candidate_skills(candidate_skills, skill_nodes)

    # score occupations based on matched skills
    scores = score_occupations(matched_skills, occupation_skill_edges)

    # rank occupations based on scores
    occupation_ranks = rank_occupations(scores, occupation_nodes, top_k=top_k)

    matched_skill_names = skill_nodes[skill_nodes['skill_idx'].isin(matched_skills)]['skill_name']

    print(f"\nCandidate ID: {candidate_id}")

    print(f"Candidate Name: {candidate_name}\n")
    print("Matched Skills:")
    for skill in matched_skill_names:
        print(f"- {skill}")
    print()
    print("Unmatched Skills:")
    for skill in unmatched_skills:
        print(f"- {skill}")
    if len(unmatched_skills) == 0:
        print("all skills matched!")
    print()

    print(f"Top {top_k} Occupations:")
    for row in occupation_ranks.itertuples():
        print(f"{(row.Index + 1):5}. {row.occupation_title:50} {row.total_score:.2f}")

if __name__ == "__main__":
    main()



