import subprocess

def run_pipeline():
    print("Starting training pipeline...")
    subprocess.run(["python", "-m", "src.train_model"])
    print("Training completed")

if __name__ == "__main__":
    run_pipeline()