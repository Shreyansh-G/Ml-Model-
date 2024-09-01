from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def normalize_coordinates(coordinates):
    scaler = StandardScaler()
    normalized_coordinates = scaler.fit_transform(coordinates)
    return normalized_coordinates, scaler

def cluster_hierarchical(coordinates, distance_threshold=None, n_clusters=None):
    coordinates = np.array(coordinates)
    normalized_coordinates, _ = normalize_coordinates(coordinates)

    if distance_threshold is not None and n_clusters is not None:
        raise ValueError("Exactly one of distance_threshold or n_clusters should be set, not both.")
    if distance_threshold is None and n_clusters is None:
        raise ValueError("Either distance_threshold or n_clusters must be set.")

    if distance_threshold is not None:
        hierarchical = AgglomerativeClustering(
            distance_threshold=distance_threshold, 
            linkage='ward', 
            n_clusters=None
        )
    else:
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage='ward'
        )

    labels = hierarchical.fit_predict(normalized_coordinates)

    clusters = {}
    for idx, label in enumerate(labels):
        label = int(label)  # Convert numpy.int64 to int
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({
            "latitude": coordinates[idx][0],
            "longitude": coordinates[idx][1]
        })

    return clusters, labels

def extract_coordinates(data):
    coordinates = []
    for route in data.get('routes', []):
        for leg in route.get('legs', []):
            for point in leg.get('points', []):
                coordinates.append([point['latitude'], point['longitude']])
    return coordinates

@app.route('/cluster', methods=['POST'])
def cluster():
    data = request.json
    coordinates = extract_coordinates(data)

    # Run Hierarchical Clustering
    clusters, _ = cluster_hierarchical(coordinates, distance_threshold=1.52)

    return jsonify(clusters)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




