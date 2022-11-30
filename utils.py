import os

HOME = os.getcwd()
DATASET_LOCATION = os.path.join(HOME, 'dataset', 'formatted')

class colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'

    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_bar(char='*', length=100):
    print(char * length)

