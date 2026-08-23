from huggingface_hub import HfApi

api = HfApi()

# Create the repo if it doesn't exist
api.create_repo(
    repo_id="RudreshDutta/legal-summarizer",
    repo_type="model",
    exist_ok=True,  # do nothing if it already exists
)

# Upload the model folder
api.upload_folder(
    folder_path="model",
    repo_id="RudreshDutta/legal-summarizer",
    repo_type="model",
)