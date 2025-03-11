# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
import cython
from libc.math cimport exp, sqrt, sin
import matplotlib.pyplot as plt
import time

# Type definitions
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# Constants
cdef double PI = np.pi

def main(int n_particles=100, bint benchmark_mode=False):
    """ Optimized main loop using Cython """
    # Parameters
    cdef int n = n_particles
    cdef double dt = 0.02
    cdef int nt = 100
    cdef int nt_setup = 400
    cdef int n_out = 25
    cdef double b = 4
    cdef double m = 1.0 / n
    cdef double h = 40.0 / n
    cdef double t = 0.0
    
    # Initialize particles
    cdef np.ndarray[DTYPE_t, ndim=2] x = np.linspace(-3.0, 3.0, n).reshape(-1, 1)
    cdef np.ndarray[DTYPE_t, ndim=2] u = np.zeros((n, 1), dtype=DTYPE)
    
    # Simulation variables
    cdef np.ndarray[DTYPE_t, ndim=2] rho, P, a, u_mhalf
    cdef int i, j
    
    # Start timing
    start_time = time.time()
    
    # Initial calculations
    rho = density(x, m, h)
    P = pressure(x, rho, m, h)
    a = acceleration(x, u, m, rho, P, b, h)
    u_mhalf = u - 0.5 * dt * a
    
    # Main loop
    for i in range(-nt_setup, nt):
        # Leapfrog integration
        u_phalf = u_mhalf + a * dt
        x = x + u_phalf * dt
        u = 0.5 * (u_mhalf + u_phalf)
        u_mhalf = u_phalf.copy()
        
        if i >= 0:
            t += dt
        
        if i == -1:
            u = np.ones_like(u)
            u_mhalf = u.copy()
            b = 0
        
        # Update physics
        rho = density(x, m, h)
        P = pressure(x, rho, m, h)
        a = acceleration(x, u, m, rho, P, b, h)
    
    # End timing
    exec_time = time.time() - start_time
    
    if benchmark_mode:
        return exec_time
    else:
        # Plotting (keep original Python code here)
        pass

def benchmark():
    """ Run benchmark for different particle numbers and plot results """
    # Benchmark parameters
    particle_numbers = [10, 20, 40, 80, 160, 320, 640, 1280]  # Different particle counts to test
    runs = 5  # Number of runs per particle count
    
    # Storage for results
    avg_times = []
    std_devs = []
    
    print("Starting benchmark...")
    
    for n in particle_numbers:
        print(f"\nRunning benchmark for n = {n} particles")
        run_times = []
        
        for run in range(runs):
            print(f"  Run {run + 1}/{runs}...", end=" ", flush=True)
            # Run simulation with current particle count
            run_time = main(n_particles=n, benchmark_mode=True)
            run_times.append(run_time)
            print(f"{run_time:.2f}s")
        
        # Calculate statistics for this particle count
        avg_time = np.mean(run_times)
        std_dev = np.std(run_times)
        
        avg_times.append(avg_time)
        std_devs.append(std_dev)
        
        print(f"  n = {n}: Avg = {avg_time:.2f}s, Std Dev = {std_dev:.2f}s")
    
    # Plot benchmark results
    plt.figure(figsize=(10, 6))
    plt.errorbar(particle_numbers, avg_times, yerr=std_devs, 
                 fmt='-o', capsize=5, capthick=2, elinewidth=2)
    plt.xlabel('Number of Particles')
    plt.ylabel('Average Execution Time (s)')
    plt.title('Performance Benchmark (Cython)')
    plt.grid(True)
    plt.savefig('cython_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nBenchmark complete! Results saved to cython_benchmark_results.png")

@cython.boundscheck(False)
@cython.wraparound(False)
def density(np.ndarray[DTYPE_t, ndim=2] x, double m, double h):
    cdef int n = x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] rho = np.zeros((n,1), dtype=DTYPE)
    cdef int i, j
    cdef double diff, kernel_val
    
    for i in range(n):
        for j in range(n):
            diff = x[i,0] - x[j,0]
            kernel_val = kernel(diff, h, 0)
            rho[i,0] += m * kernel_val
    return rho

@cython.boundscheck(False)
@cython.wraparound(False)
def pressure(np.ndarray[DTYPE_t, ndim=2] x, 
             np.ndarray[DTYPE_t, ndim=2] rho, 
             double m, double h):
    cdef int n = x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] drho = np.zeros((n,1), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] ddrho = np.zeros((n,1), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] P = np.zeros((n,1), dtype=DTYPE)
    cdef int i, j
    cdef double diff, deriv1, deriv2
    
    # First derivatives
    for i in range(n):
        for j in range(n):
            diff = x[i,0] - x[j,0]
            deriv1 = kernel(diff, h, 1)
            drho[i,0] += m * deriv1
    
    # Second derivatives
    for i in range(n):
        for j in range(n):
            diff = x[i,0] - x[j,0]
            deriv2 = kernel(diff, h, 2)
            ddrho[i,0] += m * deriv2
    
    # Pressure calculation
    for i in range(n):
        for j in range(n):
            diff = x[i,0] - x[j,0]
            kernel_val = kernel(diff, h, 0)
            term = 0.25 * (drho[i,0]**2 / rho[i,0] - ddrho[i,0]) * m / rho[i,0]
            P[i,0] += term * kernel_val
    
    return P

@cython.boundscheck(False)
@cython.wraparound(False)
def acceleration(np.ndarray[DTYPE_t, ndim=2] x,
                 np.ndarray[DTYPE_t, ndim=2] u,
                 double m,
                 np.ndarray[DTYPE_t, ndim=2] rho,
                 np.ndarray[DTYPE_t, ndim=2] P,
                 double b,
                 double h):
    cdef int n = x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] a = np.zeros((n,1), dtype=DTYPE)
    cdef int i, j
    cdef double diff, fac, deriv
    
    for i in range(n):
        a[i,0] = -u[i,0] * b - x[i,0]
        for j in range(n):
            if i == j:
                continue
            diff = x[i,0] - x[j,0]
            deriv = kernel(diff, h, 1)
            fac = -m * (P[i,0]/rho[i,0]**2 + P[j,0]/rho[j,0]**2)
            a[i,0] += fac * deriv
    return a

cdef inline double kernel(double r, double h, int deriv) nogil:
    """ Optimized kernel function """
    cdef double h_sq = h * h
    cdef double r_sq = r * r
    cdef double term = exp(-r_sq / h_sq) / (h * sqrt(PI))
    
    if deriv == 0:
        return term
    elif deriv == 1:
        return term * (-2 * r) / (h_sq)
    elif deriv == 2:
        return term * (4 * r_sq - 2 * h_sq) / (h_sq * h_sq)
    else:
        return term * (-8 * r*r_sq + 12 * h_sq * r) / (h_sq * h_sq * h_sq)