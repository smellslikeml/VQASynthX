import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pydicom
from presidio_image_redactor import DicomImageRedactorEngine
import time


def compare_dicom_images(
    instance_original: pydicom.dataset.FileDataset,
    instance_redacted: pydicom.dataset.FileDataset,
    figsize: tuple = (11, 11),
) -> None:
    """
    Shows DICOM pixel arrays as images for both original and redacted images.

    :param instance_original: Single DICOM instance (with PHI text).
    :param instance_redacted: Single DICOM instance (redacted PHI).
    :param figsize: Figure size in inches (width, height)
    """
    print(
        f"Comparing original and redacted images for {instance_original.SOPInstanceUID}"
    )
    # _, ax = plt.subplots(1, 2, figsize=figsize)
    # ax[0].imshow(instance_original.pixel_array, cmap="gray")
    # ax[0].set_title('Original')
    # ax[1].imshow(instance_redacted.pixel_array, cmap="gray")
    # ax[1].set_title('Redacted')
    # This function is not called in the non-interactive script.


if __name__ == "__main__":
    # Instantiate DICOM image redactor engine object
    engine = DicomImageRedactorEngine()

    # Get paths from environment variables
    input_path = os.environ.get("INPUT_DIR")
    output_parent_dir = os.environ.get("OUTPUT_DIR")

    print(f"Input Directory: {input_path}")
    print(f"Output Directory: {output_parent_dir}")

    if not input_path or not os.path.isdir(input_path) or not os.listdir(input_path):
        print("\nERROR: Input directory is empty or does not exist.")
        exit(1)

    # Redact text PHI from DICOM images
    print("Starting pixel data redaction...")
    engine.redact_from_directory(
        input_dicom_path=input_path,
        output_dir=output_parent_dir,
        fill="contrast",
        use_metadata=True,
        allow_list=[
            "[M]",
            "[F]",
            "[U]",
            "PORTABLE",
            "portable",
            "upright",
            "SEMI-ERECT",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ],
        save_bboxes=True,
    )
    print("Pixel data redaction complete.")
