"""
Example 2: Attribute Error Bug

This example demonstrates an AttributeError bug that can be minimized.
"""

# Unused import
import json
import os  # This is never used

# Unused utility functions
def format_data(data):
    """Format data for display."""
    return json.dumps(data, indent=2)

def validate_input(value):
    """Validate input value."""
    if value is None:
        return False
    return len(value) > 0

# Unused class
class DataProcessor:
    def __init__(self):
        self.data = []

    def add(self, item):
        self.data.append(item)

    def process(self):
        return [x * 2 for x in self.data]

# The bug: calling a method that doesn't exist
def trigger_bug():
    """Trigger an AttributeError."""
    obj = "string"  # Strings don't have 'append' method
    obj.append("test")  # This will fail
    return obj

# Main
def main():
    print("Starting...")
    unused = 123

    # Trigger the bug
    result = trigger_bug()
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
