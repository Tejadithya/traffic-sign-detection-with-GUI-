"""
Quick Entrypoint Launcher for Traffic Sign AI Recognition Studio
"""
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import main

if __name__ == "__main__":
    main()
