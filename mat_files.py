#open and show mat files in this directory






import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

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
        #load only if the name of the file is SwissRollData.mat
        if mat_file != 'SwissRollData.mat':
            continue

        # Load MAT file
        mat_data = scipy.io.loadmat(file_path)

        # Display or process the loaded data as needed
        print(f"Loaded data from {mat_file}:")
        print(mat_data)
        print("\n")
        print (mat_data.keys())
        print("\n")
        print (mat_data['Ct'][0][0])
        print("\n")
        print (np.min(mat_data['Ct'][0]))
        for i in range(0, 1000):
            # print (mat_data['Ct'][0][i])
        # for i in range(0, 100):
            # print (mat_data['Ct'][1][i])
            if mat_data['Ct'][1][i] +  mat_data['Ct'][0][i] != 1:
                print (mat_data['Ct'][1][i] +  mat_data['Ct'][0][i])
        #save the data to a txt file
        print("FINISHED")
        f = open("SwissRollData.txt", "w")
        f.write(str(mat_data))  

        blue_points = []
        red_points = []
        for i in range(0, mat_data['Ct'].shape[1]):
            if mat_data['Ct'][0][i] == 1:
                blue_points.append([mat_data['Yt'][0][i], mat_data['Yt'][1][i]])
            else:
                red_points.append([mat_data['Yt'][0][i], mat_data['Yt'][1][i]])
        
        #plot the data
        blue_points = np.array(blue_points)
        red_points = np.array(red_points)
        #scatter with small points
        plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue', s=1)
        plt.scatter(red_points[:, 0], red_points[:, 1], color='red',s=1)
        plt.show()
        X = mat_data['Yt'][0]
        Y = mat_data['Yt'][1]
        Colors = mat_data['Ct'][0]

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
