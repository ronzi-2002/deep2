#open and show mat files in this directory






import os
import scipy.io
import matplotlib.pyplot as plt

def load_and_display_mat_files(directory):
    # Get a list of all files in the specified directory
    files = os.listdir(directory)

    # Filter for MAT files (assuming they have a '.mat' extension)
    mat_files = [file for file in files if file.endswith('.mat')]

    if not mat_files:
        print("No MAT files found in the specified directory.")
        return

    # Loop through each MAT file and load/display its content
    for mat_file in mat_files:
        file_path = os.path.join(directory, mat_file)

        # Load MAT file
        mat_data = scipy.io.loadmat(file_path)

        # Display or process the loaded data as needed
        print(f"Loaded data from {mat_file}:")
        print(mat_data)

        # Optional: Visualize the data using matplotlib
        # Modify this part based on the structure of your data
        # For example, if mat_data contains a variable named 'data'
        # you can plot it as follows:
        # plt.plot(mat_data['data'])
        # plt.title(f"Data from {mat_file}")
        # plt.show()

if __name__ == "__main__":
    # Specify the directory containing your MAT files
    directory_path = os.getcwd()
    load_and_display_mat_files(directory_path)
