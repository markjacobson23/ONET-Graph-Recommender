import pandas as pd

def load_featured_tables(input_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    occupation_nodes = pd.read_csv(f"{input_dir}/featured_occupation_nodes.csv")
    skill_nodes = pd.read_csv(f"{input_dir}/featured_skill_nodes.csv")
    occupation_skill_edges = pd.read_csv(f"{input_dir}/occupation_skill_edges.csv")
    return occupation_nodes, skill_nodes, occupation_skill_edges

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
        .merge(occupation_nodes, on="occupation_idx")
        .sort_values("total_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    occupation_ranks = ranked_occupations[["occupation_idx", "occupation_title", "total_score"]]

    return occupation_ranks

def main():

    input_dir = "../../data/processed/tables/featured"

    candidate_skills = [
        "Programming",
        "Mathematics",
        "Critical Thinking",
        "Complex Problem Solving",
        "Systems Analysis",
    ]

    # load featured tables
    occupation_nodes, skill_nodes, occupation_skill_edges = load_featured_tables(input_dir)

    # match candidate skills to skill nodes
    matched_skills, unmatched_skills = match_candidate_skills(candidate_skills, skill_nodes)

    # score occupations based on matched skills
    scores = score_occupations(matched_skills, occupation_skill_edges)

    # rank occupations based on scores
    occupation_ranks = rank_occupations(scores, occupation_nodes)

    matched_skill_names = skill_nodes[skill_nodes['skill_idx'].isin(matched_skills)]['skill_name']
    print("Matched Skills:")
    for skill in matched_skill_names:
        print(f"- {skill}")
    print()
    print("Unmatched Skills:")
    for skill in unmatched_skills:
        print(f"- {skill}")
    print()

    print("Top 10 Occupations:")
    for row in occupation_ranks.itertuples():
        print(f"{(row.Index + 1):5}. {row.occupation_title:50} {row.total_score:.2f}")

if __name__ == "__main__":
    main()



