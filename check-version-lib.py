import importlib.metadata

packages = [
    "fastapi",
    "langchain",
    "langchain-anthropic",
    "langchain-core",
    "langchain-openai",
    "langchain-unstructured",
    "langsmith",
    "moviepy",
    "openai",
    "opencv-contrib-python",
    "opencv-python",
    "opencv-python-headless",
    "Pillow",
    "pydantic",
    "pydub",
    "pymongo",
    "python-docx",
    "python-dotenv",
    "python-magic",
    "python-multipart",
    "pytz",
    "requests",
    "scenedetect",
    "sentence-transformers",
    "unstructured",
    "unstructured-client",
    "uvicorn",
    "supabase",
    "httpx",
    "torch",
    "torchvision",
    "torchaudio",
    "xformers",
    "accelerate",
    "diffusers",
    "sageattention",
    "transformers",
    "sentencepiece",
    "av",
    "numpy",
    "scipy",
    "torchsde",
    "einops",
    "safetensors",
    "google-genai",
    "peft",
]

for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} not installed")
