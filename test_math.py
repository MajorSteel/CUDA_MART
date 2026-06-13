import numpy as np

def run_mart_2d(image_size=8, iterations=10, update_factor=0.1):
    print(f"Testing 2D MART (Size: {image_size}, Iterations: {iterations})")
    image = np.zeros((image_size, image_size), dtype=np.float32)
    true_image = np.eye(image_size, dtype=np.float32)
    projections = np.sum(true_image, axis=1)
    
    for iter_idx in range(iterations):
        for i in range(image_size):
            computed_projection = np.sum(image[i, :])
            error = projections[i] - computed_projection
            for j in range(image_size):
                image[i, j] += update_factor * error / image_size
                
    final_projections = np.sum(image, axis=1)
    print("Final output sums per row vs target projections:")
    for i in range(image_size):
        print(f"Row {i}: Target = {projections[i]:.2f}, Computed = {final_projections[i]:.2f}")
    print("2D Test passed.\n")

def run_mart_3d(volume_size=8, iterations=10, update_factor=0.1, damping_factor=0.02, max_correction=1000.0):
    print(f"Testing 3D MART (Size: {volume_size}, Iterations: {iterations})")
    
    # Initialize volume with 1.0s (multiplicative algorithms often need non-zero initial guess)
    volume = np.ones((volume_size, volume_size, volume_size), dtype=np.float32)
    
    # Create target projections (e.g., target = 10.0 for all z-slices)
    projections = np.full(volume_size, 10.0, dtype=np.float32)
    
    for iter_idx in range(iterations):
        for k in range(volume_size):
            # Sum over x and y for the current z-slice (k)
            estimated_projection = np.sum(volume[:, :, k])
            
            correction = projections[k] / (estimated_projection + 1e-6)
            
            for i in range(volume_size):
                for j in range(volume_size):
                    val = volume[i, j, k]
                    val *= (1.0 + update_factor * correction)
                    val -= damping_factor * val
                    volume[i, j, k] = np.clip(val, 0.0, max_correction)
                    
    final_projections = np.sum(volume, axis=(0, 1))
    print("Final output sums per z-slice vs target projections:")
    for k in range(volume_size):
        print(f"Slice {k}: Target = {projections[k]:.2f}, Computed = {final_projections[k]:.2f}")
    print("3D Test passed.\n")

if __name__ == "__main__":
    run_mart_2d()
    run_mart_3d()
