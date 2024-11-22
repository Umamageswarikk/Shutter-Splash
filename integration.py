import os

def main():
    while True:
        print("\n=== 360 Photo Booth Menu ===")
        print("1. Plain View")
        print("2. Full Frame")
        print("3. Square Frame")
        print("4. With VIPs")
        print("5. Exit")

        try:
            choice = int(input("\nEnter your choice (1-5): "))
        except ValueError:
            print("Invalid input! Please enter a number between 1 and 5.")
            continue

        # Map choice to corresponding Python files
        scripts = {
            1: "main01.py",
            2: "main.py",
            3: "main2.py",
            4: "main3.py"
        }

        if choice in scripts:
            script_to_run = scripts[choice]
            if os.path.exists(script_to_run):
                print(f"Running {script_to_run}...")
                os.system(f"python {script_to_run}")
            else:
                print(f"Error: {script_to_run} not found!")
        elif choice == 5:
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid choice! Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
