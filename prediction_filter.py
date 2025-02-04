import numpy as np
from filterpy.kalman import KalmanFilter

class PredictionFilter:
    def __init__(self, initial_state, window_size=5, distance_threshold=200, warm_up_count=10):
        self.window = np.tile(initial_state, (window_size, 1))
        self.window_size = window_size
        self.distance_threshold = distance_threshold
        self.warm_up_count = warm_up_count
        self.prediction_count = 0
        
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, dx, dy], Measurement: [x, y]
        self.kf.x = np.array([initial_state[0], initial_state[1], 0, 0])
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 1000
        self.kf.R = np.eye(2) * 100  # Increased measurement noise
        self.kf.Q = np.eye(4) * 0.1

    def update(self, new_point):
        self.prediction_count += 1
        
        # Kalman filter prediction and update
        self.kf.predict()
        self.kf.update(new_point)
        
        # Get the Kalman filter estimate
        kf_estimate = self.kf.x[:2]
        
        # During warm-up phase, accept points within a wider range
        if self.prediction_count <= self.warm_up_count:
            if 450 <= new_point[0] <= 550 and 450 <= new_point[1] <= 550:
                self.window = np.vstack((self.window[1:], new_point))
                return new_point
            else:
                print(f"Dropped outlier during warm-up: {new_point}")
                return None
        
        # After warm-up, use more stringent filtering
        # Check if the new point is too far from the Kalman estimate
        if np.linalg.norm(new_point - kf_estimate) > self.distance_threshold:
            print(f"Dropped outlier (distance): {new_point}")
            return None
        
        # Update window
        self.window = np.vstack((self.window[1:], new_point))
        
        # Calculate weighted average (more weight to recent predictions)
        weights = np.linspace(0.2, 1, self.window_size) # need to update this.
        weighted_avg = np.average(self.window, axis=0, weights=weights)
        
        return weighted_avg