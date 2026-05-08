import json
def load_candidate_profile(profile_path):
    with open(profile_path, "r") as file:
        data = json.load(file)
        return data["candidate_id"],data["candidate_name"] , data["skills"]
print(load_candidate_profile("profiles/mark_v1.json"))