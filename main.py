from medvision.config import DEVICE

from scripts.train import main as train


if __name__ == "__main__":

    print("MedVision - Complex-Valued MLP-Mixer")

    print("Running on:", DEVICE)

    train()