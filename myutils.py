import sys

def printnow(s):
    """Write string to stdout and flush immediately"""
    sys.stdout.write(s + "\n")
    sys.stdout.flush()