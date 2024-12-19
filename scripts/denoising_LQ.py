import numpy as np
import cv2
import os

def degrade(img, deg_type, param=15):
    """
    Add degradation to an image based on the specified type.

    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        deg_type (str): Degradation type, e.g., 'noisy'.
        param (int): Parameter for degradation, e.g., noise level.

    Returns:
        (Numpy array): Degraded image, shape (h, w, c), range [0, 1], float32.
    """
    if deg_type == 'noisy':
        return add_gaussian_noise(img, sigma=param)
    else:
        raise ValueError(f"Unsupported degradation type: {deg_type}")

def add_gaussian_noise(img, sigma=10, clip=True):
    """
    Add Gaussian noise to an image.

    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        sigma (float): Noise level (measured in range 255).
        clip (bool): Whether to clip pixel values to [0, 1].

    Returns:
        (Numpy array): Noisy image.
    """
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma / 255.0
    noisy_img = img + noise
    if clip:
        noisy_img = np.clip(noisy_img, 0, 1)
    return noisy_img

def is_image_file(filename):
    """
    Check if a file is an image based on its extension.

    Args:
        filename (str): File name.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)

def generate_LQ(sourcedir, savedir, deg_type='noisy', param=50):
    """
    Generate degraded images and save them to a directory.

    Args:
        sourcedir (str): Path to the source directory containing images.
        savedir (str): Path to the directory to save degraded images.
        deg_type (str): Type of degradation to apply.
        param (int): Parameter for degradation, e.g., noise level.

    Returns:
        None
    """
    if not os.path.isdir(sourcedir):
        raise FileNotFoundError(f"Source directory not found: {sourcedir}")

    os.makedirs(savedir, exist_ok=True)
    filepaths = [f for f in os.listdir(sourcedir) if is_image_file(f)]

    for i, filename in enumerate(filepaths):
        print(f"No.{i + 1} -- Processing {filename}")
        try:
            # Read image
            image = cv2.imread(os.path.join(sourcedir, filename)) / 255.0

            # Apply degradation
            image_LQ = (degrade(image, deg_type, param) * 255).astype(np.uint8)

            # Save degraded image
            cv2.imwrite(os.path.join(savedir, filename), image_LQ)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print('Finished processing all images!')

if __name__ == "__main__":
    # Define source and save directories
    sourcedir = "/home/ubuntu/Image-restoration/CycleRDM/Figs"
    savedir = "/home/ubuntu/Image-restoration/CycleRDM/Figs_LQ"

    # Generate degraded images
    generate_LQ(sourcedir, savedir, deg_type='noisy', param=50)
