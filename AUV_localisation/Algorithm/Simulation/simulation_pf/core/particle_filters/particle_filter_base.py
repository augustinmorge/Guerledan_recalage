from abc import abstractmethod
import copy
import numpy as np
from scipy.linalg import sqrtm

class ParticleFilter:
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
        * Abstract class
    """

    def __init__(self, number_of_particles, limits, process_noise, measurement_noise):
        """
        Initialize the abstract particle filter.

        :param number_of_particles: Number of particles
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax]
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular]
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle]
        """

        if number_of_particles < 1:
            print("Warning: initializing particle filter with number of particles < 1: {}".format(number_of_particles))

        # Initialize filter settings
        self.n_particles = number_of_particles
        self.particles = []

        # State related settings
        self.state_dimension = 3  # x, y, theta
        self.x_min = limits[0]
        self.x_max = limits[1]
        self.y_min = limits[2]
        self.y_max = limits[3]
        self.z_min = limits[4]
        self.z_max = limits[5]
        self.v_min = limits[6]
        self.v_max = limits[7]

        # Set noise
        self.Q = process_noise
        self.R = measurement_noise

    def initialize_particles_uniform(self):
        """
        Initialize the particles uniformly over the world assuming a 3D state (x, y, heading). No arguments are required
        and function always succeeds hence no return value.
        """

        # Initialize particles with uniform weight distribution
        weight = 1.0 / self.n_particles
        WEIGHT = weight*np.ones((self.n_particles,1))
        self.particles = [WEIGHT,\
                       np.vstack((np.random.uniform(self.x_min, self.x_max, size=(1,self.n_particles)),\
                                  np.random.uniform(self.y_min, self.y_max, size=(1,self.n_particles)),\
                                  np.random.uniform(self.z_min, self.z_max, size=(1,self.n_particles)),\
                                  np.random.uniform(self.v_min, self.v_max, size=(1,self.n_particles)),\
                                  np.random.uniform(self.v_min, self.v_max, size=(1,self.n_particles)),\
                                  np.random.uniform(self.v_min, self.v_max, size=(1,self.n_particles))))]

    def get_average_state(self):
        """
        Compute average state according to all weighted particles

        :return: Average x-position, y-position and orientation
        """

        # Compute weighted average
        avg_x = np.sum(self.particles[0]/np.sum(self.particles[0])*self.particles[1])
        avg_y = np.sum(self.particles[0]/np.sum(self.particles[0])*self.particles[2])
        avg_z = np.sum(self.particles[0]/np.sum(self.particles[0])*self.particles[3])
        avg_vx = np.sum(self.particles[0]/np.sum(self.particles[0])*self.particles[4])
        avg_vy = np.sum(self.particles[0]/np.sum(self.particles[0])*self.particles[5])
        avg_vz = np.sum(self.particles[0]/np.sum(self.particles[0])*self.particles[6])

        return [avg_x, avg_y, avg_z, avg_vx, avg_vy, avg_vz]

    def normalize_weights(weighted_samples):
        """
        Normalize all particle weights.
        """

        # Compute sum weighted samples

        sum_weights = np.sum(self.particles[0])

        # Check if weights are non-zero
        if sum_weights < 1e-15:
            print("Weight normalization failed: sum of all weights is {} (weights will be reinitialized)".format(sum_weights))

            # Set uniform weights
            return 1.0 / len(weighted_samples) * np.ones((len(weighted_sample),1))

        # Return normalized weights
        return weighted_sample / sum_weights


    def validate_state(self, state):
        """
        Validate the state. State values outide allowed ranges will be corrected for assuming a 'cyclic world'.

        :param state: Input particle state.
        :return: Validated particle state.
        """

        # Make sure state does not exceed allowed limits (cyclic world)

        # while (state[0] < self.x_min).all:
        #     state[0] += (self.x_max - self.x_min)
        # while (state[0] > self.x_max).all:
        #     state[0] -= (self.x_max - self.x_min)
        #
        # while (state[1] < self.y_min).all:
        #     state[1] += (self.y_max - self.y_min)
        # while (state[1] > self.y_max).all:
        #     state[1] -= (self.y_max - self.y_min)
        #
        # while (state[2] < self.z_min).all:
        #     state[2] += (self.z_max - self.z_min)
        # while (state[2] > self.z_max).all:
        #     state[2] -= (self.z_max - self.z_min)
        #
        # while (state[3:] < self.v_min).all:
        #     state[3:] += (self.v_max - self.v_min)
        # while (state[3:] > self.v_max).all:
        #     state[3:] -= (self.v_max - self.v_min)

        return state


    def propagate_samples(self, samples, U=np.array([[0],[0],[0]])):
        """
        Propagate samples

        :param samples: Samples (unweighted particles) that must be propagated
        :U: Acceleration of the IMU
        :return: propagated samples
        """

        #A CHANGER
        dt = 0.1

        F = np.array([[1,0,0,dt,0,0],
                      [0,1,0,0,dt,0],
                      [0,0,1,0,0,dt],
                      [0,0,0,1,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,1]])

        B = np.array([[0,0,0],
                      [0,0,0],
                      [0,0,0],
                      [dt,0,0],
                      [0,dt,0],
                      [0,0,dt]])

        propagate_samples = copy.deepcopy(samples)

        propagate_samples = F@propagate_samples
        propagate_samples += B@U
        propagate_samples += sqrtm(self.Q)@np.random.normal(0, 1, size=(len(propagate_samples),self.n_particles))

        # Make sure we stay within cyclic world
        return self.validate_state(propagate_samples)


    def compute_likelihood(self, sample, measurement, landmarks):
        """
        Compute likelihood p(z|sample) for a specific measurement given sample state and landmarks.

        :param sample: Sample (unweighted particle) that must be propagated
        :param measurement: List with measurements, for each landmark [distance_to_landmark, angle_wrt_landmark], units
        are meters and radians
        :param landmarks: Positions (absolute) landmarks (in meters)
        :return Likelihood
        """

        # Initialize measurement likelihood
        likelihood_sample = 1.0
        speed = [0.]
        # Loop over all landmarks for current particle

        def h(sample):
            #Rajouter la vitesse plus tard
            H = np.zeros((len(landmarks),1))
            for i, lm in enumerate(landmarks):
                dx = sample[0,i] - lm[0]
                dy = sample[1,i] - lm[1]
                dz = sample[2,i] - lm[2]
                expected_distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                H[i] = expected_distance
            return(H)

            """
            R = [1] #A CHANGER
            speed = R@speed
            H[i-3] = measurement[len(measurement)-3]
            H[i-2] = measurement[len(measurement)-2]
            H[i-1] = measurement[len(measurement)-1]
            """

        print("->",measurement-h(sample),"<-")
        print(np.linalg.inv(self.R))

        # Return importance weight based on all landmarks
        return np.exp(-1/2*(measurement - h(sample)).T@np.linalg.inv(self.R)@(measurement - h(sample)))

    @abstractmethod
    def update(self, robot_forward_motion, robot_angular_motion, measurements, landmarks):
        """
        Process a measurement given the measured robot displacement. Abstract method that must be implemented in derived
        class.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """

        pass
