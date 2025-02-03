import numpy as np

def generate_valuations(n_objects, base_range=(-10, 10), theta=0.0):
    """
    Generates valuation vectors v1 and v2 for n_objects.
    theta controls the alignment:
      - theta = 0: v2 = v1 (max competitive)
      - theta = pi: v2 = -v1 (max cooperative)
      - Intermediate values yield mixed settings.
    """
    v1 = np.random.uniform(base_range[0], base_range[1], n_objects)
    random_vector = np.random.randn(n_objects)
    if np.linalg.norm(v1) > 0:
        projection = (np.dot(random_vector, v1) / np.linalg.norm(v1)**2) * v1
        ortho = random_vector - projection
    else:
        ortho = random_vector
    if np.linalg.norm(ortho) > 0:
        ortho = ortho / np.linalg.norm(ortho) * np.linalg.norm(v1)
    else:
        ortho = np.zeros_like(v1)
    v2 = np.cos(theta) * v1 + np.sin(theta) * ortho
    return v1, v2

if __name__ == '__main__':
    n = 4
    v1, v2 = generate_valuations(n, theta=0.0)
    print("v1:", v1)
    print("v2:", v2)
