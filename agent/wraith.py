#!/usr/bin/env python3
import sys
import os

# Add the current directory to sys.path so we can run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.main import app

if __name__ == "__main__":
    app()
