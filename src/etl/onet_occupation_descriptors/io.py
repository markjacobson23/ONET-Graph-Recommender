
def save_occupation_descriptor_edges(occupation_descriptor_edges, output_dir, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    occupation_descriptor_edges.to_csv(output_dir / filename, index=False)

def save_occupation_nodes(occupation_nodes, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    occupation_nodes.to_csv(output_dir / "occupation_nodes.csv", index=False)

def save_descriptor_nodes(descriptor_nodes, output_dir, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_nodes.to_csv(output_dir / filename, index=False)


