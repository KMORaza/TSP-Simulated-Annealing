import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the distance or cost matrix
def create_distance_matrix(n):
    # Generate a random distance matrix (symmetric)
    np.random.seed(42)
    matrix = np.random.rand(n, n)
    np.fill_diagonal(matrix, 0)  # no self-loops
    matrix = (matrix + matrix.T) / 2  # make symmetric
    return matrix

# Calculate total tour length
def tour_length(matrix, tour):
    n = len(tour)
    total_length = 0
    for i in range(n):
        j = (i + 1) % n  # next city in the tour
        total_length += matrix[tour[i], tour[j]]
    return total_length

# Simulated Annealing function
def simulated_annealing(matrix, initial_temperature=1000, cooling_rate=0.95, stopping_temperature=1e-5):
    n = matrix.shape[0]
    current_state = list(range(n))  # initial tour: [0, 1, ..., n-1]
    random.shuffle(current_state)   # random initial tour
    current_cost = tour_length(matrix, current_state)

    temperature = initial_temperature
    history = [(current_state[:], current_cost)]  # to store history of solutions

    while temperature > stopping_temperature:
        new_state = current_state[:]
        # Generate new state by swapping two cities
        i, j = sorted(random.sample(range(n), 2))
        new_state[i:j+1] = reversed(new_state[i:j+1])
        new_cost = tour_length(matrix, new_state)

        # Acceptance probability
        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temperature):
            current_state = new_state
            current_cost = new_cost

        # Cool the temperature
        temperature *= cooling_rate

        # Append to history
        history.append((current_state[:], current_cost))

    return history

# Function to update plot for animation
def update_plot(frame, history, line, text):
    tour, cost = history[frame]
    line.set_data(*zip(*[points[i] for i in tour]))
    text.set_text(f'Temperature: {len(history) - frame - 1:.2f}\nCost: {cost:.2f}')
    return line, text

# Example usage for animation
if __name__ == "__main__":
    num_cities = 10
    distance_matrix = create_distance_matrix(num_cities)
    history = simulated_annealing(distance_matrix)

    # Prepare city coordinates for plotting
    points = np.random.rand(num_cities, 2)  # random city coordinates

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_title('Traveling Salesman Problem - Simulated Annealing')

    # Plot initial tour
    init_tour, init_cost = history[0]
    line, = ax.plot(*zip(*[points[i] for i in init_tour]), marker='o', linestyle='-')
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # Create animation
    ani = animation.FuncAnimation(fig, update_plot, frames=len(history), fargs=(history, line, text),
                                  interval=200, blit=True, repeat=False)

    plt.show()
