from huggingface_hub import HfApi
import os

api = HfApi()

token = os.getenv("HF_TOKEN")
repo_id = "btwitsPratyush/neuroscan-ai"

if not token:
    print("❌ HF_TOKEN environment variable not set. Run: export HF_TOKEN=your_token")
else:
    print("Uploading backend files to Hugging Face...")
    api.upload_folder(
        folder_path=".",
        path_in_repo=".",
        repo_id=repo_id,
        repo_type="space",
        token=token,
        allow_patterns=["server/*", "models/*", "Dockerfile", "requirements.txt", ".dockerignore"]
    )
    print("Done!")
