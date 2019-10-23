import sys
from xib.families import get_families, get_all_distances

if __name__ == "__main__":
    root = get_families(sys.argv[1])
    print(get_all_distances())
