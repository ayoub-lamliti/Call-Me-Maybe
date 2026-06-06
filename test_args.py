import argparse


arg = argparse.ArgumentParser(description="test")
arg.add_argument("--number", type=int, help="enter number", default="0")

args = arg.parse_args()

print(args.number * 2)