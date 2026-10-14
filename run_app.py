import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.app.app import main

if __name__ == '__main__':
    main()
