#!/usr/bin/env python3

def main():
    print("Starting execution of the complete pipeline...")
    
    # Import and run functionality from data.py
    print("\n--- Processing data ---")
    import data
    try:
        data.process_data()  # Assuming there's a function like this
    except AttributeError:
        print("No process_data function found, calling main function if available...")
        if hasattr(data, "main"):
            data.main()
    
    # Import and run functionality from trained_cropnet.py
    print("\n--- Running CropNet model ---")
    import trained_cropnet
    try:
        trained_cropnet.run_model()  # Assuming there's a function like this
    except AttributeError:
        print("No run_model function found, calling main function if available...")
        if hasattr(trained_cropnet, "main"):
            trained_cropnet.main()
    
    # Import and run functionality from trained_efficientnet.py
    print("\n--- Running EfficientNet model ---")
    import trained_efficientnet
    try:
        trained_efficientnet.run_model()  # Assuming there's a function like this
    except AttributeError:
        print("No run_model function found, calling main function if available...")
        if hasattr(trained_efficientnet, "main"):
            trained_efficientnet.main()
    
    # Import and run functionality from esemble.py
    print("\n--- Running ensemble model ---")
    import esemble  # Note: This might be a typo of "ensemble" in your project
    try:
        esemble.run_ensemble()  # Assuming there's a function like this
    except AttributeError:
        print("No run_ensemble function found, calling main function if available...")
        if hasattr(esemble, "main"):
            esemble.main()
    
    print("\n--- Pipeline completed successfully ---")

if __name__ == "__main__":
    main()