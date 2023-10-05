import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a trained autoencoder model from previous steps
# and a function reconstruction_error to compute errors

def with_threshold(errors, threshold_percentile, df, test_indices):
    
   # 2. Set a Threshold for Anomaly Detection
    threshold = np.percentile(errors, threshold_percentile)  # Setting threshold at the given percentile
    anomalies = np.where(errors > threshold)

    # 3. Visualize Reconstruction Errors

    # Histogram of reconstruction errors
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75, label='Reconstruction Errors')
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold:.2f})')
    plt.title('Histogram of Reconstruction Errors')
    plt.xlabel('Error')
    plt.ylabel('Number of Incidents')
    plt.legend()
    plt.show()

    # Get the 'number' feature values for the anomalies
    anomaly_indices = anomalies[0]
    anomaly_data = df[['number', 'short_description', 'description', 'assignment_group']].iloc[test_indices[anomaly_indices]]

    return anomaly_data
