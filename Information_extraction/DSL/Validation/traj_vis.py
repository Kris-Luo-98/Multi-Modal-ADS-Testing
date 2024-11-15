import os
import yaml
import matplotlib.pyplot as plt

# Folder containing YAML files
DSL_folder = r"C:\Users\Kris\Desktop\Multi-Modal-ADS-Testing\Information_extraction\DSL\Encoded_2024-11-04_00-13-48"

# Folder to save visualizations
output_folder = r"C:\Users\Kris\Desktop\Multi-Modal-ADS-Testing\Information_extraction\DSL\Validation\traj_vis_res"
os.makedirs(output_folder, exist_ok=True)

# Iterate over all YAML files in the DSL folder
for filename in os.listdir(DSL_folder):
    if filename.endswith(".yaml"):
        data_id = filename.split(".")[0]

        # Load the YAML data
        with open(os.path.join(DSL_folder, filename), "r") as file:
            data = yaml.safe_load(file)

        # Initialize the plot
        plt.figure()

        # Plot each trajectory if present in the YAML data
        actors_data = data.get("Actors", {})
        for key, traj_str in actors_data.items():
            if "traj" in key:  # Only process keys with "traj" in their name
                traj = eval(traj_str)  # Convert string to list of tuples
                x, y = zip(*traj)  # Separate x and y coordinates
                plt.plot(x, y, marker='o', label=key)  # Plot trajectory with markers

        # Set plot labels and title
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title(f"Trajectories for Scenario {data_id}")
        plt.legend()

        # Save the plot
        output_path = os.path.join(output_folder, f"{data_id}.png")
        plt.savefig(output_path)
        plt.close()  # Close the plot to free memory

print("Visualization files created in", output_folder)
