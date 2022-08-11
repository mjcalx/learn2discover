import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__)) 
try:
    sys.path.index(current_path)
except ValueError:
    sys.path.append(current_path)

def main():
    learn_to_discover = Learn2Discover()
    learn_to_discover.run()


class Learn2Discover:
    """
    Learn2Discover
    """

    def __init__(self):
        pass
    def run(self):
        pass

if __name__=="__main__":
    main()