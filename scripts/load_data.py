from datasets import load_dataset
import yaml

def load_hf_dataset(dataset_name="timdettmers/openassistant-guanaco"):
    """Load a dataset from HuggingFace Hub."""
    dataset = load_dataset(dataset_name)
    return dataset

if __name__ == "__main__":
    # Load config (e.g., "medical", "legal" from datasets.yaml)
    with open("configs/datasets.yaml") as f:
        config = yaml.safe_load(f)
    
    dataset = load_hf_dataset(config["dataset_name"])
    dataset.save_to_disk("data/processed/")
    print(f"✅ Downloaded & saved {config['dataset_name']} to data/processed/")
