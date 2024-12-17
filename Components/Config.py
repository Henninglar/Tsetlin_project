import argparse

def default_args(**kwargs):
    # Defines the parsers that collets the arguments. Below areguments are added,
    parser = argparse.ArgumentParser()
    # THE FOLLOWING argument names with the default values are added to the parser.
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--number-of-clauses", default=100, type=int)
    parser.add_argument("--T", default=80, type=int)
    parser.add_argument("--s", default=1.2, type=float)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--hypervector-size", default=1024, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=1024, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument("--number-of-examples", default=50000, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=169, type=int)

    args = parser.parse_args()  # Collects all default config values above.
    return args  # Returns all the values above so that we can use in in the main code.