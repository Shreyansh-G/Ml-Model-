from flask import Flask, request, jsonify
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

app = Flask(__name__)

def normalize_coordinates(coordinates):
    scaler = StandardScaler()
    normalized_coordinates = scaler.fit_transform(coordinates)
    return normalized_coordinates, scaler

def estimate_gmm_components(coordinates, max_components=10):
    normalized_coordinates, _ = normalize_coordinates(coordinates)

    best_n_components = 1
    best_score = -1

    for n_components in range(1, max_components + 10):
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        labels = gmm.fit_predict(normalized_coordinates)

        try:
            score = silhouette_score(normalized_coordinates, labels)
            if score > best_score:
                best_score = score
                best_n_components = n_components
        except ValueError:
            pass

    return best_n_components

def cluster_gmm(coordinates, n_components):
    coordinates = np.array(coordinates)
    normalized_coordinates, scaler = normalize_coordinates(coordinates)

    gmm = GaussianMixture(n_components=n_components, random_state=0)
    labels = gmm.fit_predict(normalized_coordinates)

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

    # Estimate the number of components for GMM
    best_n_components = estimate_gmm_components(coordinates, max_components=8)

    # Run GMM clustering
    clusters, _ = cluster_gmm(coordinates, n_components=best_n_components)

    return jsonify(clusters)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
