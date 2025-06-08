import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns
import glob
import os
import shutil

############################################################
### READING TOPO FILES
############################################################
# Directory wih the topo files to simualte
topo_dir = "./TOPOS/"
# Getting the list of all the topo files (files ending with .topo extension)
topo_list = glob.glob(os.path.join(topo_dir, "*.topo"))
print(topo_list)
print(f"Simulating {len(topo_list)} topo files...")


# Reading the topo files and converting them into adjacecy matrices
def parse_topo(topo_path):
    # Reading the topo file as a dataframe using pandas
    topo_df = pd.read_csv(topo_path, sep=r"\s+")
    # print(topo_df)
    # Replace 2s with -1s - required for boolean ising
    topo_df["Type"] = topo_df["Type"].replace({2: -1})
    # print(topo_df)
    # Using netowrkxx to load the topo file as a directed graph
    topo_net = nx.from_pandas_edgelist(
        topo_df,
        source="Source",
        target="Target",
        edge_attr="Type",
        create_using=nx.DiGraph,
    )
    # Getting the node list in the same roder s it will be in the adjacency matrix
    nodes = topo_net.nodes()
    # print(nodes)
    # Converting back to an adjacency matrix
    topo_adj = nx.to_numpy_array(topo_net, nodelist=nodes, weight="Type")
    # print(topo_adj)
    return topo_adj, list(nodes)


############################################################


############################################################
# Simulation Related Functions
############################################################
# Function to generate random inital conditions for simulation
def gen_initalconds(num_nodes, num_initconds=100):
    initcod_array = np.random.randint(0, 2, size=(num_initconds, num_nodes))
    return initcod_array


# Function for simulation of a newtork over one intial condition
def ising_simulate_sync(adjmat, initcond, num_steps):
    # Storing the states
    state_list = []
    # Append the intial condition to the state list
    state_list.append(initcond)
    # Keep running the simulation till the end of the number of steps specified
    for step in range(num_steps):
        # print(step)
        # Calling the Sync update for the step
        new_state = adjmat @ state_list[-1]
        # print(new_state)
        # Update mask
        update_mask = new_state != 0
        # Flipping the state values of the nodes based on if they are greater or lesser than 0
        new_state[update_mask] = np.where(new_state[update_mask] > 0, 1, -1)
        # Assigning the previous values to node which are 0
        new_state[~update_mask] = state_list[-1][~update_mask]
        # print(new_state)
        # Append to the state list
        state_list.append(new_state)
    return state_list


# Function for async simulation of a network over one initial condition
def ising_simulate_async(adjmat, initcond, num_steps):
    # Storing the states
    state_list = []
    # Append the initial condition to the state list
    state_list.append(initcond)
    # Keep running the simulation till the end of the number of steps specified
    for step in range(num_steps):
        # Take current state
        current_state = state_list[-1].copy()
        # Pick one random node to update
        node_to_update = np.random.randint(len(current_state))
        # Compute input sum for that node
        input_sum = adjmat[node_to_update, :] @ current_state
        # Apply the same logic as sync version:
        if input_sum > 0:
            new_value = 1
        elif input_sum < 0:
            new_value = -1
        else:
            # If input sum == 0, retain previous value
            new_value = current_state[node_to_update]
        # Update the selected node
        current_state[node_to_update] = new_value
        # Append the new state to state_list
        state_list.append(current_state)
    return state_list


# Function to simulate all the initial conditions for a particualar update mode
def ising_simualte(adjmat, initcond_array, node_list, num_steps, update_mode):
    # DataFrame to store all the individual runs
    simul_df = []
    # RUn the simualtions and append the datafames to the list to concatenate later
    for initcond_num, initcond in enumerate(initcond_array):
        if update_mode == "async":
            state_df = ising_simulate_async(topo_adj, initcond, num_steps)
        elif update_mode == "sync":
            state_df = ising_simulate_sync(topo_adj, initcond, num_steps)
        else:
            raise ValueError("Unkown Update Type.")
        # Convert the state list to pandas dataframe
        state_df = pd.DataFrame(state_df, columns=node_list)
        # Add the steps column
        state_df["Step"] = range(1, len(state_df) + 1)
        # Add the initial condition number column
        state_df["InitCondNum"] = initcond_num + 1
        # Replace -1s with 0s
        state_df[node_list] = state_df[node_list].replace(-1, 0)
        # Append to the simulate dataframe
        simul_df.append(state_df)
    # Concatenate the dataframes into a single dataframe
    simul_df = pd.concat(simul_df, axis=0)
    # Return the dataframe
    return simul_df


############################################################


############################################################
# Simulation Analysis Functions
############################################################
# Function to get the state transition graph (STG) from the simulation resutls
def get_stg(simul_result):
    # Make sure all the states are integers and not float
    simul_result[node_list] = simul_result[node_list].astype(int)
    # Make the state column
    simul_result["State"] = simul_result[node_list].astype(str).agg("".join, axis=1)
    # print(simul_result)
    # Placeholder to accumulate all the edges fir the STG
    stg = []
    # Group by the Inital condition and get the STG newtork as an edge list
    for init_cond, group in simul_result.groupby("InitCondNum"):
        # Sorting by the step value (sanity check - not needed)
        group = group.sort_values("Step").reset_index(drop=True)
        # Make the STG df
        group_stg = group[["State", "InitCondNum", "Step"]].copy()
        # Add the next state column
        group_stg.loc[:, "Next_State"] = group_stg["State"].shift(-1)
        # print(group_stg)
        # Removing the last row as it has None
        group_stg = group_stg[:-1]
        # # Reroder the columns
        # group_stg = group_stg[["State", "Next_State", "Step", "InitCondNum"]]
        # Appending to the stg list
        stg.append(group_stg[["State", "Next_State"]])
        # print(group_stg)
    # Concatenate the individual stgs
    stg = pd.concat(stg, axis=0)
    # Get counts of the unique edges
    stg = stg.groupby(["State", "Next_State"]).size().reset_index(name="Count")
    # Normliase the counts but the number of steps
    stg["Count"] = stg["Count"] / stg["Count"].sum()
    return stg


# Function to plot the state transition graphs
def plot_stg(stg, update_mode, topo_name, result_dir):
    # Converting to a networkx directed newtork
    stg = nx.from_pandas_edgelist(
        stg,
        source="State",
        target="Next_State",
        edge_attr="Count",
        create_using=nx.DiGraph(),
    )
    # Plotting the STG
    # Choose layout
    # pos = nx.spring_layout(stg, seed=42)
    pos = nx.circular_layout(stg)
    # Plot
    plt.figure(figsize=(8, 6))
    nx.draw(
        stg,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=1000,
        font_size=14,
        arrowsize=20,
    )
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(stg, "Count")
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    # Adding the edge labels
    nx.draw_networkx_edge_labels(stg, pos, edge_labels=edge_labels, font_size=12)
    # Adding the title
    plt.title(f"State Transition Graph (STG) - {topo_name}")
    plt.axis("off")
    plt.savefig(
        os.path.join(result_dir, topo_name, f"{topo_name}_STG_{update_mode}.png"),
        dpi=300,
    )
    return None


############################################################

# Creeating a directory to store the results
result_dir = "./IsingResults"
os.makedirs(
    result_dir, exist_ok=True
)  # Will not rais an error if the folder exists already

# Creating a directory to store the plots
plot_dir = "./STG_Plots/"
os.makedirs(
    plot_dir, exist_ok=True
)  # Will not rais an error if the folder exists already

# Main loop to simualte and store results
for topo_path in topo_list:
    # Extract the topo name
    topo_name = os.path.basename(topo_path).replace(".topo", "")
    print(f"Simulating: {topo_name}")
    # Create a folder in the result_dir with the topo name
    os.makedirs(os.path.join(result_dir, topo_name), exist_ok=True)
    # Parsing the topo file
    topo_adj, node_list = parse_topo(topo_path)
    print(f"Adjacency Matrix:\n {topo_adj}")
    print(f"Node List:\n {node_list}")
    # Generating the inital conditions
    init_conds = gen_initalconds(len(node_list), num_initconds=100)
    # print(f"Inital Condition Matrix:\n {init_conds}")
    print(f"Shape of Inital Condition Matrix: {init_conds.shape}")
    # Simulation of the network over all the intial conditions in sync and async modes
    for update_mode in ["sync", "async"]:
        # state_list = ising_simulate_async(topo_adj, init_conds[1], num_steps=10)
        simul_result = ising_simualte(
            topo_adj, init_conds, node_list, num_steps=10, update_mode=update_mode
        )
        print(simul_result)
        # Saving the results as a csv file
        simul_result.to_csv(
            os.path.join(result_dir, topo_name, f"{topo_name}_{update_mode}.csv"),
            index=False,
        )
        # Getting the state transition graph
        stg = get_stg(simul_result)
        print(stg)
        # Saving the STG
        stg.to_csv(
            os.path.join(result_dir, topo_name, f"{topo_name}_STG_{update_mode}.csv"),
            index=False,
        )
        # PLotting the state transition graphs
        plot_stg(stg, update_mode, topo_name, result_dir)
        # plt.show()
        # break

# Moving all the images to the plots directory
[shutil.copy(f, "./STG_Plots/") for f in glob.glob("./IsingResults/*/*.png")]
