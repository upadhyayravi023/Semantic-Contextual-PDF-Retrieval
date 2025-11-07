import logging
from colorama import Fore, Style, init

# Initialize colorama for Windows/Unix
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

log = logging.getLogger(__name__)