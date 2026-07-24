

# Unused import

  # This is never used

# Unused utility functions




# Unused class


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
