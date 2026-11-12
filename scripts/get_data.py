"""Get each dataset into data/raw/<dataset> and check it. See deepfake_detector/data/download.py.

python scripts/get_data.py disk
python scripts/get_data.py ffpp --script faceforensics_download_v4.py
python scripts/get_data.py celebdf --zip Celeb-DF-v2.zip
python scripts/get_data.py dfdc --zip dfdc_train_part_0.zip --n 1000
python scripts/get_data.py check
"""

import sys

from deepfake_detector.data.download import main

if __name__ == "__main__":
    sys.exit(main())
