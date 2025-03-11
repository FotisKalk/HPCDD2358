#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np
import time

def main(n_particles=100, benchmark_mode=False):
    # Parameters (use n_particles instead of hardcoded n)
    n = n_particles
    dt = 0.02
    nt = 100
    nt_setup = 400
    n_out = 25
    b = 4
    m = 1/n
    h = 40/n
    t = 0.

    # Initialize plot only if not in benchmark mode
    if not benchmark_mode:
        plt.figure()
        xx_plot = np.linspace(-4.0, 4.0, 400).reshape(-1,1)
        plt.plot(xx_plot, 0.5*xx_plot**2, linewidth=5, color=[0.7, 0.7, 0.9])

    # Initialize
    x = np.linspace(-3.0, 3.0, num=n).reshape(-1,1)
    u = np.zeros((n,1))
    
    start_time = time.time()
    
    # Simulation loop
    rho = density(x, m, h)
    P = pressure(x, rho, m, h)
    a = acceleration(x, u, m, rho, P, b, h)
    u_mhalf = u - 0.5 * dt * a

    for i in range(-nt_setup, nt):
        u_phalf = u_mhalf + a*dt
        x = x + u_phalf*dt
        u = 0.5*(u_mhalf+u_phalf)
        u_mhalf = u_phalf
        
        if i >= 0:
            t += dt
            
        if i == -1:
            u = np.zeros((n,1)) + 1.0
            u_mhalf = u
            b = 0

        rho = density(x, m, h)
        P = pressure(x, rho, m, h)
        a = acceleration(x, u, m, rho, P, b, h)

        # Plotting inside the loop but only when not in benchmark mode
        if not benchmark_mode and (i >= 0) and (i % n_out == 0):
            xx = np.linspace(-4.0, 4.0, 400).reshape(-1,1)
            rr = probeDensity(x, m, h, xx)
            rr_exact = 1./np.sqrt(np.pi) * np.exp(-(xx-np.sin(t))**2/2.)**2
            plt.plot(xx, rr_exact, linewidth=2, color=[.6, .6, .6])
            plt.plot(xx, rr, linewidth=2, color=[1.*i/nt, 0, 1.-1.*i/nt], 
                    label=f'$t={t:.2f}$')

    exec_time = time.time() - start_time
    
    if benchmark_mode:
        return exec_time
    else:
        # Finalize plot
        plt.legend()
        plt.xlabel('$x$')
        plt.ylabel('$|\psi|^2$')
        plt.axis([-2, 4, 0, 0.8])
        plt.savefig(f'solution_{n:.0f}.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
def benchmark():
    # Benchmark parameters
    particle_numbers = [10, 20, 40, 80, 160, 320, 640, 1280] 
    runs = 5  # Number of runs per particle count
    
    # Storage for results
    avg_times = []
    std_devs = []
    
    print("Starting benchmark...")
    
    for n in particle_numbers:
        print(f"\nRunning benchmark for n = {n} particles")
        run_times = []
        
        for run in range(runs):
            print(f"  Run {run+1}/{runs}...", end=" ", flush=True)
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
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(particle_numbers, avg_times, yerr=std_devs, 
                 fmt='-o', capsize=5, capthick=2, elinewidth=2)
    plt.xlabel('Number of particles')
    plt.ylabel('Average execution time (s)')
    plt.title('Initial performance benchmark')
    plt.grid(True)
    
    # Save and show plot
    plt.savefig('benchmark_results_original.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nBenchmark complete! Results saved to benchmark_results.png")
   
   
   
   
def kernel(r, h, deriv):
  """ SPH Gaussian smoothing kernel (1D).
  Input: distance r, scaling length h, derivative order deriv
  Output: weight
  """
  return {
    '0': h**-1 / np.sqrt(np.pi) * np.exp(-r**2/h**2),
    '1': h**-3 / np.sqrt(np.pi) * np.exp(-r**2/h**2) * (-2*r),
    '2': h**-5 / np.sqrt(np.pi) * np.exp(-r**2/h**2) * ( 4*r**2 - 2*h**2),
    '3': h**-7 / np.sqrt(np.pi) * np.exp(-r**2/h**2) * (-8*r**3 + 12*h**2*r)
  }[deriv]
     
   
   
def density(x, m, h):
  """ Compute density at each of the particle locations using smoothing kernel
  Input: positions x, SPH particle mass m, scaling length h
  Output: density
  """
  
  n = x.size
  rho = np.zeros((n,1))
  
  for i in range(0, n):
    # calculate vector between two particles
    uij = x[i] - x
    # calculate contribution due to neighbors
    rho_ij = m*kernel( uij, h, '0' )
    # accumulate contributions to the density
    rho[i] = rho[i] + np.sum(rho_ij)
    
  return rho


 
def pressure(x, rho, m, h):
  """Compute ``pressure'' at each of the particles using smoothing kernel
  P = -(1/4)*(d^2 rho /dx^2 - (d rho / dx)^2/rho)
  Input: positions x, densities rho, SPH particle mass m, scaling length h
  Output: pressure
  """
  
  n = x.size
  drho = np.zeros((n,1))
  ddrho = np.zeros((n,1))
  P = np.zeros((n,1))
  
  # add the pairwise contributions to 1st, 2nd derivatives of density
  for i in range(0, n):
    # calculate vector between two particles
    uij = x[i] - x
    # calculate contribution due to neighbors
    drho_ij  = m * kernel( uij, h, '1' )
    ddrho_ij = m * kernel( uij, h, '2' )
    # accumulate contributions to the density
    drho[i]  = np.sum(drho_ij)
    ddrho[i] = np.sum(ddrho_ij)

  # add the pairwise contributions to the quantum pressure
  for i in range(0, n):
    # calculate vector between two particles
    uij = x[i] - x
    # calculate contribution due to neighbors
    P_ij = 0.25 * (drho**2 / rho - ddrho) * m / rho * kernel( uij, h, '0' )
    # accumulate contributions to the pressure
    P[i] = np.sum(P_ij)

  return P
 
 
 
def acceleration( x, u, m, rho, P, b, h):
  """ Calculates acceletaion of each particle due to quantum pressure, harmonic potential, velocity damping
  Input: positions x, velocities u, SPH particle mass m, densities rho, pressure P, damping coeff b, scaling length h
  Output: accelerations
  """
	
  n = x.size
  a = np.zeros((n,1))

  for i in range(0, n):
    
    # damping & harmonic potential (0.5 x^2)
    a[i] = a[i] - u[i]*b - x[i]

    # quantum pressure (pairwise calculation)
    x_js = np.delete(x,i)
    P_js = np.delete(P,i)
    rho_js = np.delete(rho,i)
    # first, calculate vector between two particles
    uij = x[i] - x_js
    # calculate acceleration due to pressure
    fac = -m * (P[i]/rho[i]**2 + P_js/rho_js**2)
    pressure_a = fac * kernel( uij, h, '1' )
    # accumulate contributions to the acceleration
    a[i] = a[i] + np.sum(pressure_a)

  return a



def probeDensity(x, m, h, xx):
  """ Probe the density at arbitrary locations
  Input: positions x, SPH particle mass m, scaling length h, probe locations xx
  Output: density at evenly spaced points
  """	

  nxx  = xx.size
  rr = np.zeros((nxx,1))

  n = x.size

  # add the pairwise contributions to density
  for i in range(0, nxx):
      # calculate vector between two particles
      uij = xx[i] - x
      # calculate contribution due to neighbors
      rho_ij = m * kernel( uij, h, '0' )
      # accumulate contributions to the density
      rr[i] = rr[i] + np.sum(rho_ij)

  return rr




if __name__ == "__main__":
    if '--benchmark' in sys.argv:
        benchmark()
    else:
        main()