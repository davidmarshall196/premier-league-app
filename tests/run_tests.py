import pytest
import os

# Path to the directory one level above
test_dir = os.path.join(os.path.dirname(__file__), '..')

# Execute pytest tests
pytest.main([test_dir, "-p", "no:warnings", "--disable-warnings"])

if __name__ == "__main__":
    pass