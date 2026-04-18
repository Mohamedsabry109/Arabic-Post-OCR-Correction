import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def safe_load_image(img_path: str, force_flip: bool = False) -> Image.Image:
    """Safely loads an image, handling transparency, EXIF, and manual flipping."""
    image = Image.open(img_path)
    
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
    return image

def test_tiff_rendering(tiff_path: str):
    """Loads a TIFF raw vs processed and displays them side-by-side."""
    if not os.path.exists(tiff_path):
        print(f"Error: Could not find file at {tiff_path}")
        return

    # Load raw (what happens if you just do a basic open)
    raw_img = Image.open(tiff_path).convert("RGB")
    
    # Load processed (handling EXIF and Alpha)
    processed_img = safe_load_image(tiff_path, force_flip=False)
    
    # Load processed + Forced Flip (in case the dataset itself is mirrored)
    flipped_img = safe_load_image(tiff_path, force_flip=True)

    # Setup matplotlib figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.canvas.manager.set_window_title("TIFF Debugger")

    axes[0].imshow(raw_img)
    axes[0].set_title("1. Raw .convert('RGB')\n(Notice background/orientation)")
    axes[0].axis('off')

    axes[1].imshow(processed_img)
    axes[1].set_title("2. Safe Load\n(Fixed Transparency & EXIF)")
    axes[1].axis('off')

    axes[2].imshow(flipped_img)
    axes[2].set_title("3. Safe Load + Force Flip\n(Use if Arabic is still mirrored)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ---> REPLACE THIS with the path to one of your .tiff files
    TEST_FILE_PATH = "D:\Masters\Arabic-Post-OCR-Correction\data\ocr-raw-data\PATS_A01_Dataset\A01-Arial\Arial_1.tif" 
    
    test_tiff_rendering(TEST_FILE_PATH)