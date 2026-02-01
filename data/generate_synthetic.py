import os
import numpy as np

def generate_synthetic_pems(num_samples=1000, timesteps=12, num_nodes=325, features=1):
    """
    Generate synthetic dataset similar to PEMS-BAY.
    - num_samples: number of sequences
    - timesteps: history length (12 = 1 hour of 5-min intervals)
    - num_nodes: number of traffic sensors (~325)
    - features: number of features (default = 1, speed)
    """

    # Traffic speeds between 0–100 mph
    X = np.random.uniform(low=0, high=100, size=(num_samples, timesteps, num_nodes, features))

    # Targets: next timesteps (future speeds), add noise ±5 mph
    Y = X[:, -timesteps//2:, :, :] + np.random.normal(0, 5, size=(num_samples, timesteps//2, num_nodes, features))

    # Clip values to realistic range [0, 100]
    Y = np.clip(Y, 0, 100)

    return X, Y

def save_dataset(path="data/synthetic_pems_bay.npz"):
    X, Y = generate_synthetic_pems()
    np.savez_compressed(path, x=X, y=Y)
    print(f"✅ Synthetic dataset saved at {path}")
    print("X shape:", X.shape)
    print("y shape:", Y.shape)
    print("Speed range in X:", X.min(), "to", X.max())
    print("Speed range in y:", Y.min(), "to", Y.max())

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    save_dataset()