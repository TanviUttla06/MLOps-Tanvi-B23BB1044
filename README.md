# Assignment 3: End-to-End Hugging Face Model Training & Docker Deployment

Hugging Face Model Link: [tanvi130506/distilbert-imdb-mlops](https://huggingface.co/tanvi130506/distilbert-imdb-mlops)  
GitHub Repository Link: [\[Insert your GitHub URL here\]](https://github.com/TanviUttla06/MLOps-Tanvi-B23BB1044/tree/Assignment---3)

## 📌 Project Overview
This project demonstrates a complete Machine Learning Operations (MLOps) pipeline. It involves fine-tuning a transformer model for binary sentiment classification on the IMDB dataset, pushing the artifacts to the Hugging Face Hub, and containerizing the entire workflow (both training and production evaluation) using Docker with GPU acceleration.

## 🗂️ Repository Structure
* `src/` - Contains the modularized Python scripts (`train.py`, `evaluate_from_hub.py`, `data.py`, `utils.py`).
* `Dockerfile` - Development environment used for training the model.
* `Dockerfile.eval` - Production-ready, lightweight image that pulls the model from the Hub and runs evaluation.
* `requirements.txt` - Project dependencies (PyTorch, Transformers, Datasets, etc.).
* `README.md` - Project documentation and report.

---

## 🚀 Docker Build & Run Instructions

### Training Environment (Tasks 2 & 5)
Build and run the container used to train the model locally:
```bash
docker build -t mlops-train .
docker run --gpus all -it mlops-train

####  Production Evaluation Environment (Task 9)
Build and run the lightweight production image. Upon startup, this container automatically pulls the fine-tuned model from the Hugging Face Hub and evaluates it against the test dataset using GPU acceleration.

Bash
docker build -t mlops-eval -f Dockerfile.eval .
docker run --gpus all -it mlops-eval


## 1. Model Selection

Selected Model: `distilbert-base-uncased`

Reasoning: DistilBERT was chosen because it provides an optimal balance between performance and computational efficiency. It is created through knowledge distillation, making it 40% smaller and 60% faster than BERT-base while retaining 97% of its language understanding capabilities.  This makes it highly suitable for containerized MLOps workflows where minimizing Docker image size (the model weights are only 268 MB) and reducing training/inference time are key priorities.

## 2. Training Summary

The model was fine-tuned for binary sentiment classification using the IMDB Dataset. The original Jupyter Notebook was refactored into modular Python scripts and executed inside a GPU-enabled Docker container (`nvidia/cuda` base).

Training Hardware & Configuration:

API: Hugging Face `Trainer` API
Epochs: 1
Hardware: NVIDIA GPU (`--gpus all`)

Training Results:

Total Training Time: 909.26 seconds (~15 minutes)
Training Speed: 27.50 samples/second
Final Training Loss: 0.2893
Total Steps: 1,563


## 3. Evaluation Comparison

To verify the integrity of the uploaded model (Task 8), an evaluation was run locally immediately after training, and a second evaluation was run by pulling the model directly from the Hugging Face Hub into the container.

| Metric | Local Evaluation (Task 6) | Hub Evaluation (Task 8) |
| --- | --- | --- |
| Accuracy | 90.82% | 90.92% |
| F1 Score | 0.9080 | 0.9087 |
| Eval Loss | 0.2309 | 0.2302 |

Conclusion:The metrics from the Hugging Face Hub pull are consistent with the local training results (the +0.10% variance is within expected tolerances for evaluation batching). This confirms the model weights, config, and tokenizer were successfully serialized and deployed to the registry.

## 4. Challenges & Resolutions

During the development and deployment process, several technical challenges were addressed:

Hugging Face Authentication & Missing Git:

Challenge: The standard huggingface-cli login and push_to_hub methods failed because the Docker container lacked a git binary installation, resulting in `FileNotFoundError: [Errno 2] No such file or directory: 'git'`.
Resolution: Bypassed the Git credential helper by using the Hugging Face Python API (`HfApi.upload_folder`), which successfully uploaded the artifacts via direct HTTP requests.


Container Script Management:
Issue: Needed to run the Hub evaluation script (`evaluate_from_hub.py`) inside the active training container without stopping and rebuilding the entire image.
Resolution: Utilized `docker cp` to push the new script into the running `task8_container` and executed it via `docker exec -it`, drastically speeding up the debugging and verification loop.



Large CUDA/PyTorch Dependencies:

Challenge: Installing torch with GPU support in the production image required downloading over 2.5 GB of NVIDIA/CUDA libraries, drastically slowing down build times.

Resolution: Maintained the --no-cache-dir flag to ensure the final image size wasn't bloated by temporary files, and utilized Docker's layer caching so subsequent builds would trigger instantly after the initial download phase.


