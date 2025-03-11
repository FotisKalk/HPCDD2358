#!/usr/bin/env python3
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# Colab setup
assert torch.cuda.is_available(), "Enable GPU in Colab: Runtime > Change runtime type > GPU"
device = torch.device('cuda')
torch.set_default_tensor_type(torch.cuda.DoubleTensor)

def main(n_particles=100, benchmark_mode=False):
    # Parameters
    n = n_particles
    dt = 0.02
    nt = 100
    nt_setup = 400
    n_out = 25
    b = 4
    m = torch.tensor(1/n, device=device)
    h = torch.tensor(40/n, device=device)
    t = torch.tensor(0.0, device=device)

    # Initialize plot if not in benchmark mode
    if not benchmark_mode:
        plt.figure()
        xx_plot = torch.linspace(-4.0, 4.0, 400, device=device).unsqueeze(1)
        plt.plot(xx_plot.cpu(), 0.5*xx_plot.cpu()**2, linewidth=5, color=[0.7, 0.7, 0.9])

    # Initialize tensors
    x = torch.linspace(-3.0, 3.0, n, device=device).unsqueeze(1)
    u = torch.zeros((n, 1), device=device)

    # Timing setup
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()

    # Simulation loop
    rho = density(x, m, h)
    P = pressure(x, rho, m, h)
    a = acceleration(x, u, m, rho, P, b, h)
    u_mhalf = u - 0.5 * dt * a

    for i in range(-nt_setup, nt):
        u_phalf = u_mhalf + a * dt
        x = x + u_phalf * dt
        u = 0.5 * (u_mhalf + u_phalf)
        u_mhalf = u_phalf.clone()
        
        if i >= 0:
            t += dt
            
        if i == -1:
            u = torch.ones_like(u)
            u_mhalf = u.clone()
            b = 0

        rho = density(x, m, h)
        P = pressure(x, rho, m, h)
        a = acceleration(x, u, m, rho, P, b, h)

        # Plotting inside the loop but only when not in benchmark mode
        if not benchmark_mode and (i >= 0) and (i % n_out == 0):
            xx = torch.linspace(-4.0, 4.0, 400, device=device).unsqueeze(1)
            rr = probeDensity(x, m, h, xx)
            rr_exact = 1./np.sqrt(np.pi) * torch.exp(-(xx-torch.sin(t))**2/2.)**2
            plt.plot(xx.cpu(), rr_exact.cpu(), linewidth=2, color=[.6, .6, .6])
            plt.plot(xx.cpu(), rr.cpu(), linewidth=2, color=[1.*i/nt, 0, 1.-1.*i/nt], 
                    label=f'$t={t.item():.2f}$')

    # End timing
    end_event.record()
    torch.cuda.synchronize()
    exec_time = start_event.elapsed_time(end_event) / 1000  # Convert ms to seconds
    
    if benchmark_mode:
        return exec_time
    else:
        # Finalize plot
        plt.legend()
        plt.xlabel('$x$')
        plt.ylabel('$|\psi|^2$')
        plt.axis([-2, 4, 0, 0.8])
        plt.savefig(f'solution_{n:.0f}_gpu.pdf', bbox_inches='tight', pad_inches=0)
        plt.show()

def benchmark():
    # Benchmark parameters
    particle_numbers = [10, 20, 40, 80, 160, 320, 640, 1280]  # Different particle counts to test
    runs = 5  # Number of runs per particle count
    
    # Storage for results
    avg_times = []
    std_devs = []
    
    print("Starting GPU benchmark...")
    
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
    
    # Plot benchmark results
    plt.figure(figsize=(10, 6))
    plt.errorbar(particle_numbers, avg_times, yerr=std_devs, 
                 fmt='-o', capsize=5, capthick=2, elinewidth=2)
    plt.xlabel('Number of Particles')
    plt.ylabel('Average Execution Time (s)')
    plt.title('GPU Performance Benchmark')
    plt.grid(True)
    plt.savefig('gpu_benchmark_results.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def kernel(r, h, deriv):
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=device))
    base = torch.exp(-(r**2)/(h**2)) / (h * sqrt_pi)
    
    if deriv == 0:
        return base
    elif deriv == 1:
        return base * (-2 * r) / h**2
    elif deriv == 2:
        return base * (4 * r**2 - 2 * h**2) / h**4
    return base * (-8 * r**3 + 12 * h**2 * r) / h**6

def density(x, m, h):
    pairwise_diff = x - x.T  # [n, n]
    kernel_vals = kernel(pairwise_diff, h, 0)
    return m * kernel_vals.sum(dim=1, keepdim=True)

def pressure(x, rho, m, h):
    pairwise_diff = x - x.T  # [n, n]
    
    drho = m * kernel(pairwise_diff, h, 1).sum(dim=1, keepdim=True)
    ddrho = m * kernel(pairwise_diff, h, 2).sum(dim=1, keepdim=True)
    
    term = 0.25 * (drho**2 / rho - ddrho) * m
    rho_reciprocal = (1 / rho).T
    P_contribution = term * rho_reciprocal
    
    kernel_0 = kernel(pairwise_diff, h, 0)
    P_total = (P_contribution * kernel_0).sum(dim=1, keepdim=True)
    
    return P_total

def acceleration(x, u, m, rho, P, b, h):
    pairwise_diff = x - x.T  # [n, n]
    kernel_1 = kernel(pairwise_diff, h, 1)
    
    P_ratio = P / rho**2
    P_j = P_ratio.T
    fac = -m * (P_ratio + P_j)
    pressure_a = (fac * kernel_1).sum(dim=1, keepdim=True)
    
    a = -u * b - x + pressure_a
    return a

def probeDensity(x, m, h, xx):
    pairwise_diff = xx - x.T  # [400, n]
    kernel_vals = kernel(pairwise_diff, h, 0)
    return m * kernel_vals.sum(dim=1, keepdim=True)

if __name__ == "__main__":
    import sys
    if '--benchmark' in sys.argv:
        benchmark()
    else:
        main()