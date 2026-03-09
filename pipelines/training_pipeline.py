import subprocess

def run_pipeline():
    print("Starting training pipeline...")
    subprocess.run(["python", "src/train_model.py"])
    print("Training completed")

if __name__ == "__main__":
    run_pipeline()