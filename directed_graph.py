"""
A directed graph is formulated, with weights assigned according to probabilities. 
Subsequently, a hierarchy of graphs is established based on percentiles. 
Connectivity is determined through well-connected components within the graph.
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

def get_top_five_predictions(image_path):
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=5)

        # Get the top five predictions for the image
        return decoded_predictions

    except Exception as e:
        print(f"Error processing image: {e}")
        return []
    
# Load pre-trained model (ResNet50)
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)

# Input: Path to a single folder
main_folder = input("Enter the path to the main folder: ")

if not os.path.isdir(main_folder):
    print("Invalid folder path.")
    exit()
    
# Create a directed graph
G = nx.DiGraph()

# Add nodes for each category
categories = ["Siamese_cat", "tabby", "Persian_cat", "Egyptian_cat",
              "Chihuahua", "American_Staffordshire_terrier", "Boston_bull", "French_bulldog", "pug",
              "Angora", "hare", "wood_rabbit", "wallaby", "hamster", "beaver", "guinea_pig", "marmot",
              "American_coot", "drake", "goose", "black_swan", "red-breasted_merganser"]
for category in categories:
    G.add_node(category)

# Dictionary to store all the predictions for each folder
all_predictions = {}
    
# Dictionary to store all the flattened predictions for each folder
folder_flattened_predictions = {}

# Dictionary to store most common predictions for each folder
folder_most_common_predictions = {}

# Iterate through each category folder
for category_folder in os.listdir(main_folder):
    category_folder_path = os.path.join(main_folder, category_folder)

    if os.path.isdir(category_folder_path):
        print(f"Processing folder: {category_folder}")

        # Get all image files in the category folder
        image_files = [os.path.join(category_folder_path, file) for file in os.listdir(category_folder_path)]
        
        # Initialize predictions list for each category
        predictions = []

        # Call the function to get top five predictions
        for image_path in image_files:
            predictions.append(get_top_five_predictions(image_path))

        # Filter out None values (cases where there's only one unique prediction)
        predictions = [prediction for prediction in predictions if prediction is not None]
        print(predictions)
                
         # Flatten the predictions list
        flattened_predictions = [item[1] for per_image_prediction in predictions for prediction in per_image_prediction for item in prediction]

        # Find the most common prediction using Counter
        most_common_prediction = Counter(flattened_predictions).most_common(1)[0][0]
        
        # Store all the flattened prediction for each folder in the dictionary
        all_predictions[category_folder] = predictions

        # Store all the flattened prediction for each folder in the dictionary
        folder_flattened_predictions[category_folder] = flattened_predictions
        
        # Store the most common prediction for the folder in the dictionary
        folder_most_common_predictions[category_folder] = most_common_prediction
        
        print(f"Most common prediction for {category_folder}: {most_common_prediction}")

# Connect nodes based on hierarchy and calculate weights
for folder, prediction in folder_most_common_predictions.items():
    for other_folder, all_other_predictions in all_predictions.items():
        weight = 0
        for other_predictions in all_other_predictions:
            for other_prediction in other_predictions[0]:
                if folder != other_folder and prediction in other_prediction:
                    try:
                        if folder_most_common_predictions[folder] == other_prediction[1]:
                            weight += other_prediction[2]
                    except Exception as e:
                        print(f"Error processing prediction: {e}")
        print(folder_most_common_predictions[folder], folder_most_common_predictions[other_folder])
        print(weight)

        # Check if the edge already exists
        if weight != 0:
            if G.has_edge(folder_most_common_predictions[folder], folder_most_common_predictions[other_folder]):
                G[folder_most_common_predictions[folder]][folder_most_common_predictions[other_folder]]["weight"] += weight
            else:
                G.add_edge(folder_most_common_predictions[folder], folder_most_common_predictions[other_folder], weight=weight)

# Extract the weights of edges
weights = [d['weight'] for u, v, d in G.edges(data=True)]

# Divide the weights into groups
sorted_weights = sorted(weights)
group_size = len(weights) // 10
remainder = len(weights) % 10
groups = []
start_index = 0

for i in range(10):
    group_length = group_size + (1 if i < remainder else 0)
    groups.append(sorted_weights[start_index:start_index + group_length])
    start_index += group_length
    
# Create a copy of the graph to manipulate
G_copy = G.copy()
percentages = 100

# Visualize the tree
pos = nx.spring_layout(G_copy, seed=42, k=5, scale=3.0)
nx.draw(G_copy, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=10, font_color='black',
        font_weight='bold', edge_color='gray')

# Annotate edges with weights
for edge in G_copy.edges(data=True):
    weight = round(edge[2]["weight"], 2) # Round to two decimal places
    plt.annotate(str(weight), xy=(pos[edge[0]] + pos[edge[1]]) / 2, color='red', ha='center', va='center')

plt.title(f"Hierarchy graph (Weight Threshold: {percentages}%)")
plt.show()
    
for group in groups:
    # Define the maximum weight based on the percentage
    percentages -= 10
    
    if percentages >= 10:
        # Remove edges with weights below the threshold
        edges_to_remove = [(u, v) for u, v, d in G_copy.edges(data=True) if d['weight'] in group]
        G_copy.remove_edges_from(edges_to_remove)

        # Visualize the tree
        pos = nx.spring_layout(G_copy, seed=42, k=5, scale=3.0)
        nx.draw(G_copy, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=10, font_color='black',
                font_weight='bold', edge_color='gray')

        # Annotate edges with weights
        for edge in G_copy.edges(data=True):
            weight = round(edge[2]["weight"], 2) # Round to two decimal places
            plt.annotate(str(weight), xy=(pos[edge[0]] + pos[edge[1]]) / 2, color='red', ha='center', va='center')

        plt.title(f"Hierarchy graph (Weight Threshold: {percentages}%)")
        plt.show()
        
        # Print the strong connected components
        components = list(nx.strongly_connected_components(G_copy))
        print("\nComponents:")
        for i, component in enumerate(components, 1):
            print(f"Component {i}: {component}")

# Initialize distance and predecessor matrices
num_nodes = len(G.nodes)
inf = float('inf')
distance_matrix = [[inf] * num_nodes for _ in range(num_nodes)]
predecessor_matrix = [[None] * num_nodes for _ in range(num_nodes)]

# Set the distance matrix based on the existing edges in the graph
for u, v, data in G.edges(data=True):
    weight = data.get('weight', 0)  # Default weight is 0 if not specified
    u_index = list(G.nodes).index(u)
    v_index = list(G.nodes).index(v)
    distance_matrix[u_index][v_index] = 4 - weight
    predecessor_matrix[u_index][v_index] = u

# Apply Floyd-Warshall algorithm with predecessor tracking
for k in range(num_nodes):
    for i in range(num_nodes):
        for j in range(num_nodes):
            if distance_matrix[i][k] + distance_matrix[k][j] < distance_matrix[i][j]:
                distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
                predecessor_matrix[i][j] = predecessor_matrix[k][j]

# Display shortest paths for each node to all others
print("Shortest Paths:")
for source_index in range(num_nodes):
    source_node = list(G.nodes)[source_index]
    print(f"* From node '{source_node}': * ")
    for target_index in range(num_nodes):
        target_node = list(G.nodes)[target_index]
        if source_node == target_node:
            print(f"To node '{target_node}': {source_node}")
        else:
            path = [target_node]
            predecessor = predecessor_matrix[source_index][target_index]
            while predecessor is not None and predecessor != source_node:
                path.insert(0, predecessor)
                predecessor_index = list(G.nodes).index(predecessor)
                predecessor = predecessor_matrix[source_index][predecessor_index]
            if predecessor == source_node:
                path.insert(0, source_node)
                print(f"To node '{target_node}': {' -> '.join(path)} (Distance: {distance_matrix[source_index][target_index]})")
            else:
                print(f"To node '{target_node}': No path exists.")

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

# Extract edge weights as features for clustering
edge_weights = 4 - np.array([d['weight'] for u, v, d in G.edges(data=True)])

# Perform Agglomerative Clustering
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
clustering = model.fit(edge_weights.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(clustering, truncate_mode="level")
plt.xlabel('Nodes')
plt.ylabel('Distance')
plt.show()

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, node_labels, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # Plot the corresponding dendrogram
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, labels=node_labels, **kwargs)

# Extract node names from the graph G and encode them into numeric identifiers
node_labels = list(G.nodes())
label_to_numeric = {label: i for i, label in enumerate(node_labels)}
numeric_labels = [label_to_numeric[label] for label in node_labels]

# Setting distance_threshold=0 ensures we compute the full tree
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
clustering = model.fit(np.array(numeric_labels).reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(clustering, node_labels=node_labels, truncate_mode="level")
plt.xlabel('Nodes')
plt.ylabel('Distance')
plt.show()
