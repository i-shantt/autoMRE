

# This function is never called and should be removed


# This class is not needed for the bug reproduction


# This function is also not called


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
