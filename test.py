import pygame
import math
import numpy as np
import time
from collections import deque

WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)
BLACK_COLOR = (0, 0, 0)

class Envir:
    def __init__(self, dim):
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)

        self.height = dim[0]
        self.width = dim[1]

        pygame.display.set_caption("Autonomous Robot Optimization")
        self.map = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont("arial", 30)
        self.text = self.font.render("default", True, self.black, self.white)
        self.textRect = self.text.get_rect()
        self.textRect.center = (dim[1] - 500, dim[0] - 100)
        self.trail_set = []

    def info(self, vx, vy, theta):
        text = f"Vx = {np.round(vx, 2)}, Vy = {np.round(vy, 2)}, Theta = {np.round(theta, 2)}"
        self.text1 = self.font.render(text, True, self.black, self.white)
        self.map.blit(self.text1, self.textRect)

    def sensor_info(self, sensor_data):
        text = f"sensor: {sensor_data}"
        self.text2 = self.font.render(text, True, self.black, self.white)
        self.textRect.center = (self.width - 850, self.height - 50)
        self.map.blit(self.text2, self.textRect)

    def trail(self, pos):
        for i in range(0, len(self.trail_set) - 1):
            pygame.draw.line(self.map, self.red,
                             (self.trail_set[i][0], self.trail_set[i][1]),
                             (self.trail_set[i + 1][0], self.trail_set[i + 1][1]))
        if self.trail_set.__sizeof__() > 30000:
            self.trail_set.pop(0)
        self.trail_set.append(pos)

    def robot_frame(self, pos, rotation):
        n = 80
        centerx, centery = pos
        x_axis = (centerx + n * math.cos(-rotation), centery + n * math.sin(-rotation))
        y_axis = (centerx + n * math.cos(-rotation + math.pi / 2),
                  centery + n * math.sin(-rotation + math.pi / 2))

        pygame.draw.line(self.map, self.blue, pos, x_axis, 3)
        pygame.draw.line(self.map, self.green, pos, y_axis, 3)

    def robot_sensor(self, pos, points):
        for point in points:
            pygame.draw.line(self.map, (0, 255, 0), pos, point)
            pygame.draw.circle(self.map, (0, 255, 0), point, 5)


class Robot:
    def __init__(self, startpos, Img, width):
        self.w = width
        self.x = startpos[0]
        self.y = startpos[1]
        self.theta = 0

        # Movement parameters (0-150 pixels/s range)
        self.vx = 0  # pixel/s
        self.vy = 0
        self.theta_d = 0  # rad/s

        # Control inputs
        self.Vgx = 0  # Global x velocity (0-150)
        self.Vgy = 0  # Global y velocity (0-150)
        self.GTheta_d = 0  # Rotation rate

        # Movement tracking for penalty calculation
        self.position_history = deque(maxlen=20)  # Stores last 20 positions
        self.rotation_history = deque(maxlen=20)  # Stores last 20 angles
        
        self.sensor_data = [0, 0, 0, 0, 0, 0]
        self.points = []
        self.crash = False
        self.time = 0
        self.cost_function = 0
        self.stationary_penalty = 0
        self.circling_penalty = 0

        # graphics
        self.img = pygame.image.load(Img)
        self.img = pygame.transform.scale(self.img, (50, 50))
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

    def draw(self, map):
        map.blit(self.rotated, self.rect)

    def move(self):
        # Simplified movement - direct application of velocities
        self.vx = self.Vgx * math.cos(self.theta) - self.Vgy * math.sin(self.theta)
        self.vy = self.Vgx * math.sin(self.theta) + self.Vgy * math.cos(self.theta)
        self.theta_d = self.GTheta_d

        # Update position and orientation
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.theta += self.theta_d * dt

        # Keep theta within -pi to pi range
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

        # Record movement history for penalty calculations
        self.position_history.append((self.x, self.y))
        self.rotation_history.append(self.theta)

        self.rotated = pygame.transform.rotozoom(self.img, math.degrees(self.theta), 1)
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

    def calculate_movement_penalties(self):
        """Calculate penalties for stationary or circling behavior"""
        self.stationary_penalty = 0
        self.circling_penalty = 0
        
        if len(self.position_history) >= 10:  # Need enough history
            # Calculate distance moved in last 10 steps
            start_pos = self.position_history[0]
            end_pos = self.position_history[-1]
            distance = math.sqrt((end_pos[0]-start_pos[0])**2 + (end_pos[1]-start_pos[1])**2)
            
            # Stationary penalty (if moved less than 5 pixels in 10 steps)
            if distance < 5:
                self.stationary_penalty = 1000 * (5 - distance)
            
            # Circling penalty (if rotating continuously without progress)
            angle_changes = [abs(self.rotation_history[i+1] - self.rotation_history[i]) 
                           for i in range(len(self.rotation_history)-1)]
            total_rotation = sum(angle_changes)
            
            if total_rotation > 4*math.pi and distance < 20:  # Full rotations without moving
                self.circling_penalty = 500 * total_rotation

    def update_sensor_data(self):
        if track_copy is None:
            return
        angles = [self.theta, np.pi/3 + self.theta, 2*np.pi/3 + self.theta, 
                np.pi + self.theta, 4*np.pi/3 + self.theta, 5*np.pi/3 + self.theta]
        edge_points = []
        edge_distances = []
        max_distance = 120  # Sensor distance limit

        for angle in angles:
            distance = 0
            edge_x, edge_y = int(self.x), int(self.y)
            
            while distance < max_distance:
                edge_x = int(self.x + distance * math.cos(angle))
                edge_y = int(self.y + distance * math.sin(angle))
                
                # Check screen boundaries
                if not (0 <= edge_x < track_copy.get_width() and 0 <= edge_y < track_copy.get_height()):
                    break
                
                # Stop if obstacle detected (black color)
                if track_copy.get_at((edge_x, edge_y)) == BLACK_COLOR:
                    break
                    
                distance += 1
            
            edge_points.append((edge_x, edge_y))
            edge_distances.append(distance)
        
        self.sensor_data = edge_distances
        self.points = edge_points

    def check_crash(self):
        edge_x, edge_y = (int(self.x), int(self.y))
        if track_copy.get_at((edge_x, edge_y)) == BLACK_COLOR:
            self.crash = True
            self.cost_function = float('inf')
        if self.time >= 20:  # 20 second time limit per iteration
            self.crash = True
            if self.cost_function < float('inf'):
                self.cost_function *= 1.2  # Slight penalty for not finishing in time

# Initialize pygame
pygame.init()
pygame.display.set_mode((713, 714))
track = pygame.image.load('vidu.png')
track_copy = track.copy()

start = (200, 60)
dims = (634, 491)

# Set to run in background
pygame.event.set_allowed([pygame.QUIT])
pygame.display.set_allow_screensaver(True)

running = True
dt = 0.01  # Time step
lasttime = pygame.time.get_ticks()
environment = Envir(dims)

number = 20
Robots = []
for i in range(number):
    Robots.append(Robot(start, "circle_blue_red_infinity.png", 1))

# APSO initialization
pop_size = number
min_max = [-3, 3]
npar = 120  # Total parameters (9x10 + 10x3 = 90 + 30 = 120)
max_iteration = 1000  # Increased from 50 to 1000

# APSO parameters with adaptive components
w_max = 0.9
w_min = 0.4
c1_max = 2.5
c1_min = 0.5
c2_max = 2.5
c2_min = 0.5

# Initialize positions and velocities
P = np.random.uniform(min_max[0], min_max[1], (pop_size, npar))
V = np.random.uniform(-1, 1, (pop_size, npar)) * 0.1

# Initialize best positions and fitness
Pbest_position = P.copy()
Pbest_fitness = np.ones(pop_size) * float('inf')
Gbest_position = np.zeros(npar)
Gbest_fitness = float('inf')

fitness_history = np.zeros(max_iteration)
robot_available = pop_size

def af(x):
    """Activation function (tanh)"""
    return np.tanh(x)

def vidu_nn(X, W, V):
    """Neural network forward pass"""
    Net = W.T @ X
    y_h = af(Net)
    output = V.T @ y_h
    return output

def adaptive_parameters(iteration, max_iter):
    """Adaptive parameter adjustment for APSO"""
    # Linearly decreasing inertia weight
    w = w_max - (w_max - w_min) * (iteration / max_iter)
    
    # Time-varying acceleration coefficients
    if iteration < max_iter / 2:
        # Exploration phase
        c1 = c1_max - (c1_max - c1_min) * (iteration / (max_iter / 2))
        c2 = c2_min + (c2_max - c2_min) * (iteration / (max_iter / 2))
    else:
        # Exploitation phase
        c1 = c1_min + (c1_max - c1_min) * ((iteration - max_iter / 2) / (max_iter / 2))
        c2 = c2_max - (c2_max - c2_min) * ((iteration - max_iter / 2) / (max_iter / 2))
    
    return w, c1, c2

iteration = 0

# Main loop
while running and iteration < max_iteration:
    # Get adaptive parameters for this iteration
    w, c1, c2 = adaptive_parameters(iteration, max_iteration)
    
    # Reset robots for new iteration
    for idx, robot in enumerate(Robots):
        robot.x = 202
        robot.y = 60
        robot.theta = np.pi/2
        robot.position_history.clear()
        robot.rotation_history.clear()
        robot.check_crash()
        robot.update_sensor_data()
        robot.crash = False
        robot.cost_function = 0
        robot.time = 0
        robot.stationary_penalty = 0
        robot.circling_penalty = 0

    robot_available = pop_size
    iteration_start_time = time.time()

    # Simulation loop for current iteration (20 seconds max)
    while robot_available > 0 and (time.time() - iteration_start_time) < 20:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not running:
            break

        # Calculate time step
        current_time = pygame.time.get_ticks()
        dt = (current_time - lasttime) / 1000
        lasttime = current_time
        
        # Limit maximum dt to prevent large jumps
        dt = min(dt, 0.1)

        # Update all robots
        for idx, robot in enumerate(Robots):
            if not robot.crash:
                # Calculate errors
                ex = 222 - robot.x
                ey = 468 - robot.y
                etheta = (-np.pi/2 - robot.theta + math.pi) % (2 * math.pi) - math.pi
                
                # Neural network input
                nn_input = np.array([[ex], [ey], [etheta], 
                                   [robot.sensor_data[0]], [robot.sensor_data[1]], 
                                   [robot.sensor_data[2]], [robot.sensor_data[3]],
                                   [robot.sensor_data[4]], [5]])
                
                # Get weights from APSO
                W = P[idx,:90].reshape(9,10)
                V1 = P[idx,90:].reshape(10,3)
                
                # Get control outputs from NN (clamped to 0-150 range)
                YY = vidu_nn(nn_input, W, V1)
                robot.Vgx = max(0, min(150, YY[0,0]))  # Clamp to 0-150
                robot.Vgy = max(0, min(150, YY[1,0]))  # Clamp to 0-150
                robot.GTheta_d = YY[2,0]

                # Calculate movement penalties
                robot.calculate_movement_penalties()
                
                # Update cost function with penalties
                robot.cost_function = robot.cost_function + \
                                      0.001*ex**2 + 0.001*ey**2 + 1000*etheta**2 + \
                                      robot.stationary_penalty + robot.circling_penalty
                
                # Move robot
                robot.move()
                robot.time = robot.time + dt
                robot.check_crash()
                
                if robot.crash:
                    robot_available -= 1
                
                # Draw robot and sensors
                robot.draw(environment.map)
                robot.update_sensor_data()
                environment.robot_frame((robot.x, robot.y), robot.theta)
                environment.robot_sensor((robot.x, robot.y), robot.points)

        # Update display
        pygame.display.update()
        environment.map.blit(track, (0, 0))

    # APSO update after each iteration
    for idx, robot in enumerate(Robots):
        J = robot.cost_function
        
        # Update personal best
        if J < Pbest_fitness[idx]:
            Pbest_fitness[idx] = J
            Pbest_position[idx] = P[idx].copy()
        
        # Update global best
        if J < Gbest_fitness:
            Gbest_fitness = J
            Gbest_position = P[idx].copy()
    
    fitness_history[iteration] = Gbest_fitness
    
    # Print iteration info with APSO parameters
    print(f"Iteration {iteration+1}/{max_iteration} - Best Fitness: {Gbest_fitness:.2f} - "
          f"Params: w={w:.2f}, c1={c1:.2f}, c2={c2:.2f} - "
          f"Time: {time.time() - iteration_start_time:.2f}s")
    
    # Update velocities and positions with adaptive parameters
    r1 = np.random.rand(pop_size, npar)
    r2 = np.random.rand(pop_size, npar)
    
    V = w * V + c1 * r1 * (Pbest_position - P) + c2 * r2 * (Gbest_position - P)
    P = P + V
    
    # Apply bounds
    P = np.clip(P, min_max[0], min_max[1])
    
    iteration += 1

    # Early stopping if we've converged
    if iteration > 50 and np.std(fitness_history[iteration-50:iteration]) < 0.1:
        print("\nEarly stopping - solution converged")
        break

# Final output
print("\nOptimization complete!")
print(f"Final best fitness: {Gbest_fitness}")
print(f"Completed iterations: {iteration}/{max_iteration}")
print("Best position weights:")
print(Gbest_position)

pygame.quit()