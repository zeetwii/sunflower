import pandas as pd
import matplotlib.pyplot as plt

from tkinter import filedialog as fd # needed to grab data file
import os # needed for file name manipulation

# Load the CSV file
fileName = fd.askopenfilename(title="Select a CSV to process")

# Replace 'your_file.csv' with the path to your CSV file
data = pd.read_csv(fileName, sep=',')

# Assuming the CSV has columns named 'Time', 'X_Velocity', 'Y_Velocity'
time = data['Time']
x_velocity = data['X']
y_velocity = data['Y']

# Plot the data
plt.figure(figsize=(30, 20))
plt.plot(time, x_velocity, label='X Velocity', color='blue', linestyle='-')
plt.plot(time, y_velocity, label='Y Velocity', color='red', linestyle='--')

# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('X and Y Velocities Over Time')
plt.legend()
plt.grid()

# Save the plot as an image
plt.savefig('velocities_plot.png')

# Show the plot (optional)
plt.show()
