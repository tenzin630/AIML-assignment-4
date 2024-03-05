import tkinter as tk
from tkinter import messagebox
import numpy as np


class KMeansGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("K-Means Clustering")
        
        self.label_k = tk.Label(master, text="Number of centroids (k):")
        self.label_k.grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.entry_k = tk.Entry(master)
        self.entry_k.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        self.objects_frame = tk.LabelFrame(master, text="Objects")
        self.objects_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
        self.centroids_frame = tk.LabelFrame(master, text="Centroids")
        self.centroids_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        self.add_object_button = tk.Button(master, text="Add Object", command=self.add_object_entry)
        self.add_object_button.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        self.add_centroid_button = tk.Button(master, text="Add Centroid", command=self.add_centroid_entry)
        self.add_centroid_button.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        self.run_button = tk.Button(master, text="Run K-Means", command=self.run_kmeans)
        self.run_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

        self.object_entries = []
        self.centroid_entries = []

    def add_object_entry(self):
        x_label = tk.Label(self.objects_frame, text="X-axis:")
        y_label = tk.Label(self.objects_frame, text="Y-axis:")
        x_label.grid(row=len(self.object_entries), column=0, padx=5, pady=2, sticky="e")
        y_label.grid(row=len(self.object_entries), column=2, padx=5, pady=2, sticky="e")

        x_entry = tk.Entry(self.objects_frame)
        y_entry = tk.Entry(self.objects_frame)
        x_entry.grid(row=len(self.object_entries), column=1, padx=5, pady=2, sticky="ew")
        y_entry.grid(row=len(self.object_entries), column=3, padx=5, pady=2, sticky="ew")
        self.object_entries.append((x_entry, y_entry))

    def add_centroid_entry(self):
        x_label = tk.Label(self.centroids_frame, text="X-axis:")
        y_label = tk.Label(self.centroids_frame, text="Y-axis:")
        x_label.grid(row=len(self.centroid_entries), column=0, padx=5, pady=2, sticky="e")
        y_label.grid(row=len(self.centroid_entries), column=2, padx=5, pady=2, sticky="e")

        x_entry = tk.Entry(self.centroids_frame)
        y_entry = tk.Entry(self.centroids_frame)
        x_entry.grid(row=len(self.centroid_entries), column=1, padx=5, pady=2, sticky="ew")
        y_entry.grid(row=len(self.centroid_entries), column=3, padx=5, pady=2, sticky="ew")
        self.centroid_entries.append((x_entry, y_entry))

    def run_kmeans(self):
        try:
            k = int(self.entry_k.get())
            m = len(self.object_entries)
            n = 2  # Since we have x and y coordinates for each object

            objects = []
            for x_entry, y_entry in self.object_entries:
                x = float(x_entry.get())
                y = float(y_entry.get())
                objects.append([x, y])

            centroids = []
            for x_entry, y_entry in self.centroid_entries:
                x = float(x_entry.get())
                y = float(y_entry.get())
                centroids.append([x, y])

            # Perform K-means clustering
            cluster_assignments, centroids = kmeans(np.array(objects), np.array(centroids))

            # Display results
            result_str = "\nCluster assignments:\n"
            for i, cluster in enumerate(cluster_assignments):
                result_str += f"Object {i + 1} belongs to cluster {int(cluster) + 1}\n"

            result_str += "\nFinal centroids:\n"
            for i, centroid in enumerate(centroids):
                result_str += f"Centroid {i + 1}: {centroid}\n"

            messagebox.showinfo("K-Means Clustering Result", result_str)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


def kmeans(objects, centroids):
    m, n = objects.shape
    k = centroids.shape[0]
    max_iterations = 100

    cluster_assignments = np.zeros(m)

    for _ in range(max_iterations):
        for i in range(m):
            distances = [np.linalg.norm(objects[i] - centroids[j]) for j in range(k)]
            cluster_assignments[i] = np.argmin(distances)

        new_centroids = np.zeros((k, n))
        counts = np.zeros(k)

        for i in range(m):
            cluster = int(cluster_assignments[i])
            new_centroids[cluster] += objects[i]
            counts[cluster] += 1

        for j in range(k):
            if counts[j] != 0:
                new_centroids[j] /= counts[j]

        if np.array_equal(new_centroids, centroids):
            break

        centroids = new_centroids

    return cluster_assignments, centroids


def main():
    root = tk.Tk()
    app = KMeansGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
