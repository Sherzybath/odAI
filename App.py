import sys

def main():
    print("Choose Overlay Mode:")
    print("1. Full Analysis Overlay")
    print("2. Quick Detection Overlay")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        import overlay_main
        overlay_main.launch()
    elif choice == "2":
        import overlay_light
        overlay_light.launch()
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
