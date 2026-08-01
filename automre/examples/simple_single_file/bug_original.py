"""
Example 1: Simple Type Error Bug

This example demonstrates a simple bug where a function is called with
the wrong type. The minimized version should only include the essential
code that triggers the error.
"""

# This function is never called and should be removed
def unused_helper(x):
    """A helper function that is never used."""
    result = x * 2
    return result + 1

# This class is not needed for the bug reproduction
class UnusedClass:
    def __init__(self, value):
        self.value = value

    def process(self):
        return self.value * 3

# This function is also not called
def another_unused_function():
    data = [1, 2, 3, 4, 5]
    total = sum(data)
    return total / len(data)

# The actual bug: calling len() on an integer
def trigger_bug():
    """This function triggers the TypeError."""
    x = 42  # x is an integer
    # This will fail: len() doesn't work on integers
    return len(x)

# Main execution
if __name__ == "__main__":
    # These lines are not needed
    print("Starting program...")
    unused_var = 123

    # This triggers the bug
    result = trigger_bug()
    print(f"Result: {result}")
