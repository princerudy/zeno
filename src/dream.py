# Re-run due to code state reset
import os
import yaml
import random
from datetime import datetime

# Updated paths for new project structure
zettels_dir = os.path.join(os.path.dirname(__file__), '../data/zettels')
graph_path = os.path.join(os.path.dirname(__file__), 'components/graph.yaml')
pulls_dir = os.path.join(os.path.dirname(__file__), '../logs/pull_requests')


# 1. Load all zettels
zettels = []
for file in os.listdir(zettels_dir):
    if file.endswith(".yaml"):
        with open(os.path.join(zettels_dir, file)) as f:
            zettels.append(yaml.safe_load(f))

# 2. Pick a random zettel to start
seed = random.choice(zettels) if zettels else None

# 3. Build dummy new zettel (this would be LLM-generated in real run)
if seed:
    new_zettel = {
        "id": f"zettel-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "title": f"Dream from {seed['title']}",
        "text": f"This is a reflective thought branching from '{seed['title']}'.",
        "tags": seed.get("tags", []) + ["dreamed"],
        "anchor": f"Reflection of {seed['id']}",
        "connects_to": [seed['id']],
        "source": "zeno_dream_loop",
        "created": datetime.utcnow().isoformat()
    }

    # Save as zettel
    new_zettel_path = os.path.join(zettels_dir, new_zettel["id"] + ".yaml")
    with open(new_zettel_path, "w") as f:
        yaml.dump(new_zettel, f)

    # Save a mock pull request
    pr = {
        "title": f"✨ New Zettel: '{new_zettel['title']}'",
        "body": [
            f"**Seed Node:** {seed['id']}",
            f"**Reason:** Reflective generation",
            f"**Generated Insight:** {new_zettel['text']}",
            f"**Suggested Link:** Connects to `{seed['id']}`",
            f"_Generated on {new_zettel['created']} by Zeno agent v0.1_"
        ]
    }
    pr_path = os.path.join(pulls_dir, f"pr_{new_zettel['id']}.yaml")
    with open(pr_path, "w") as f:
        yaml.dump(pr, f)

    result = {
        "zettel_created": new_zettel["title"],
        "zettel_path": new_zettel_path,
        "pr_path": pr_path
    }
else:
    result = {"error": "No zettels found to dream from."}

result