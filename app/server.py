"""
Load the files required for PaddleOCR before starting uvicorn workers. Not doing this will cause
all workers to download simultaneously, and some of them will then crash.
"""

from loguru import logger
from paddleocr import PaddleOCR


if __name__ == "__main__":
    # This line is essential for the next one not to cause a segfault.
    # Don't ask why, just trust the process
    logger.info("Downloading PaddleOCR files...")

    PaddleOCR(use_angle_cls=True, lang="german", show_log=False)
