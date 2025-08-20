import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import integrate
from scipy.special import iv
from tqdm import tqdm 

def plot_eigenvectors(v,nt,n_f,v_0_idx = 0, v_f_idx = 5):
    for i in range(v_0_idx,v_f_idx):#v.shape[0]):
        v_plot = np.dot(v[:,i],nt)
        n_f_v = np.dot(v[:,i],n_f)
        #v_plot = v_plot - v_plot[-1]
        v_plot = np.abs(v_plot - n_f_v)
        if i == 0:
            plt.plot(v_plot,'--',label = 'Slow Mode')
            plt.legend()
        else:
            plt.plot(v_plot)
            
def calc_dndt_controlled(k,K,n,n_f,lam,lam_p,lam_c,s_hat):
    s = np.matmul(s_hat,n - n_f)
    return np.matmul(k*(np.outer(np.ones(len(n)),n)+ K)**(-1),n) - lam*n - lam_p*n - np.matmul(lam_c,s)

def load_3d_matrix_from_csv(csv_file_path):
    matrix_slices = []
    current_slice = []
    
    with open(csv_file_path, 'r') as csv_file:
        for line in csv_file:
            if line.strip():  # Non-empty line
                if line.startswith("Depth"):
                    if current_slice:
                        matrix_slices.append(current_slice)
                        current_slice = []
                else:
                    values = line.strip().split(',')
                    current_slice.append(list(map(float, values)))  # Change int to float
    
    if current_slice:
        matrix_slices.append(current_slice)
    
    matrix_3d = np.array(matrix_slices)
    return matrix_3d

def sim_dyn(n0,t_f,k,K,lam):
    def dndt(t,n):
        output = calc_dndt(k,K,n,lam)
        return output
    t_span = [0, t_f]  # Time span to solve the differential equation
    rtol = 1e-6  # Relative tolerance for the solution
    atol = 1e-9  # Absolute tolerance for the solution
    sol = solve_ivp(dndt, t_span, n0, rtol=rtol, atol=atol, dense_output=True, method='Radau')

    # Create an interpolating function for the solution
    sol_fun = sol.sol

    # Evaluate the solution at a large number of time points for plotting
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    n_eval = sol_fun(t_eval)
    return n_eval, t_eval

def sim_dyn_controlled(n0,t_f,k,K,n_f,lam,lam_p,lam_c,s_hat):
    def dndt(t,n):
        output = calc_dndt_controlled(k,K,n,n_f,lam,lam_p,lam_c,s_hat)
        return output
    t_span = [0, t_f]  # Time span to solve the differential equation
    rtol = 1e-6  # Relative tolerance for the solution
    atol = 1e-9  # Absolute tolerance for the solution
    sol = solve_ivp(dndt, t_span, n0, rtol=rtol, atol=atol, dense_output=True, method='Radau')

    # Create an interpolating function for the solution
    sol_fun = sol.sol

    # Evaluate the solution at a large number of time points for plotting
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    n_eval = sol_fun(t_eval)
    return n_eval, t_eval
    
def evaluate_fitness_controlled(k,K,n_f,lam,lam_c,s_hat,m,t_f,g):
    dists = []
    for i in range(m):
        lam_p = np.zeros(g) 
        #idx = np.random.randint(5)
        lam_p[i] = lam_p[i]+2
        nt, t = sim_dyn_controlled(n_f,t_f,k,K,n_f,lam,lam_p,lam_c,s_hat)
        dists.append(n_f_dist(n_f,nt[:,-1]))
    return np.mean(dists)

def evaluate_fitness_controlled_norm(k,K,n_f,lam,lam_c,s_hat,m,t_f,g):
    dists = []
    for i in range(m):
        lam_p = np.zeros(g) 
        #idx = np.random.randint(5)
        lam_p[i] = lam_p[i]+2
        nt, t = sim_dyn_controlled(n_f,t_f,k,K,n_f,lam,lam_p,lam_c,s_hat)
        dists.append(n_f_dist(n_f,nt[:,-1]))
    dists_norm = []
    for i in range(m):
        lam_p = np.zeros(g) 
        #idx = np.random.randint(5)
        lam_p[i] = lam_p[i]+2
        nt, t = sim_dyn_controlled(n_f,t_f,k,K,n_f,lam,lam_p,np.zeros(lam_c.shape),np.zeros(s_hat.shape))
        dists_norm.append(n_f_dist(n_f,nt[:,-1]))
    return np.mean(dists)/np.mean(dists_norm)

def n_f_dist(n_f,n_p):
    return np.linalg.norm(n_f-n_p)/np.linalg.norm(n_f)

def calc_dndt(k,K,n,lam):
    return np.matmul(k*(np.outer(np.ones(len(n)),n)+ K)**(-1),n) - lam*n

def calc_J(k,K,n,lam):
    return k*K/(K**2 + 2*K*(np.outer(np.ones(len(n)),n)) + (np.outer(np.ones(len(n)),n))**2) - lam*np.identity(len(n))

def calc_IPR(w,v):
        IPR = np.sum((v[np.argsort(w)][-1] * np.conjugate(v[np.argsort(w)][-1]))**2)
        return IPR

def simulate_dynamics(k,K,lam,n0,thresh = 0.0001, cutoff = 500000, dt = 0.01):
    n = np.array(n0)
    g = k.shape[0]
    nt = np.zeros((g,cutoff))
    for i in range(cutoff):
        n = n + calc_dndt(k,K,n,lam)*dt
        nt[:,i] = n
        if i > 0 and np.max(np.abs((nt[:,i] - nt[:,i-1])/nt[:,i])) < thresh:
            nt = nt[:,:i+1]
            break

    return nt
def von_mises_fisher_pdf(x, kappa, mu):
    return np.exp(kappa * np.dot(mu, x)) / ((2 * np.pi) ** (len(x)/2) * iv(len(x)/2 - 1, kappa))

def generate_distributed_vectors(n,k,sigma):
    vectors = []
    mu = np.zeros(n) + 1
    mu = mu/np.linalg.norm(mu)
    vectors = sp.stats.vonmises_fisher.rvs(mu=mu, kappa = sigma,size = k)
    return vectors

def optimize_nd_single_perturbation(n=5, mode_gap = 1, df = 0):
    if df == 0:
        df = (np.zeros(n) + 1)/np.linalg.norm(np.zeros(n) + 1)
    v_1 = np.zeros(n)
    v_1[0] = 1
    df_orth = np.array(df)
    df_orth[0] = 0
    df_orth = df_orth/np.linalg.norm(df_orth)
    theta = np.arccos(np.dot(df,v_1))
    alphas_s = np.linspace(0,theta,100)
    alphas_g = np.linspace(0,theta,100)
    Jx = np.zeros((n,n))
    for i in range(n):
        Jx[i,i] = -2
    Jx[0,0]= -2/mode_gap
    results = np.zeros((len(alphas_s),len(alphas_g)))    
    for i in range(len(alphas_s)):
        for j in range(len(alphas_g)):
            s = np.cos(alphas_s[i])*v_1 + np.sin(alphas_s[i])*df_orth
            g = np.cos(alphas_g[j])*v_1 + np.sin(alphas_g[j])*df_orth
            gJxs = np.matmul(g,np.matmul(np.linalg.inv(Jx),s))
            Jxgs_inv = np.linalg.inv(Jx - np.outer(g, s))
            results[i,j] = dx_norm_efficient(df,Jxgs_inv,gJxs)
    i,j = np.argwhere(results == np.min(results))[0]
    return theta, alphas_s[i], alphas_g[j]
    
def optimize_nd_many_perturbations(n=5, mode_gap = 1, sigma = 0.1, n_perturbs = 2000,df = 0):
    if df == 0:
        df = (np.zeros(n) + 1)/np.linalg.norm(np.zeros(n) + 1)
    v_1 = np.zeros(n)
    v_1[0] = 1
    df_orth = np.array(df)
    df_orth[0] = 0
    df_orth = df_orth/np.linalg.norm(df_orth)
    theta = np.arccos(np.dot(df,v_1))
    alphas_s = np.linspace(0,theta,100)
    alphas_g = np.linspace(0,theta,100)
    Jx = np.zeros((n,n))
    for i in range(n):
        Jx[i,i] = -2
    Jx[0,0]= -2/mode_gap
    results = np.zeros((len(alphas_s),len(alphas_g)))    
    dfs = generate_distributed_vectors(n,n_perturbs,sigma)

    for i in tqdm(range(len(alphas_s))):
        for j in range(len(alphas_g)):
            s = np.cos(alphas_s[i])*v_1 + np.sin(alphas_s[i])*df_orth
            g = np.cos(alphas_g[j])*v_1 + np.sin(alphas_g[j])*df_orth
            gJxs = np.matmul(g,np.matmul(np.linalg.inv(Jx),s))
            Jxgs_inv = np.linalg.inv(Jx - np.outer(g, s))
            results[i,j] = np.mean([dx_norm_efficient(df,Jxgs_inv,gJxs) for df in dfs])
    i,j = np.argwhere(results == np.min(results))[0]
    return theta, alphas_s[i], alphas_g[j]



def simulated_annealing(k,K,n_f,lam,lam_c_0,s_hat_0,m,t_f, num_iterations, temperature, cooling_rate,g,scale = 100):
    best_s_hat = s_hat_0
    best_lam_c = lam_c_0
    best_cost = evaluate_fitness_controlled(k,K,n_f,lam,scale*best_lam_c,best_s_hat,m,t_f,g)

    current_s_hat = s_hat_0
    current_lam_c = lam_c_0
    current_cost = best_cost

    costs = []

    for iteration in tqdm(range(num_iterations)):
        costs.append(current_cost)
        i_s, j_s = np.random.randint(current_s_hat.shape[0]),np.random.randint(current_s_hat.shape[1])
        i_l, j_l = np.random.randint(current_lam_c.shape[0]),np.random.randint(current_lam_c.shape[1])
        new_s_hat = np.array(current_s_hat)
        new_s_hat[i_s,j_s] = new_s_hat[i_s,j_s] + np.random.normal(0, 0.2)
        new_lam_c = np.array(current_lam_c)
        new_lam_c[i_l, j_l] = new_lam_c[i_l, j_l] + np.random.normal(0,0.2)


        # Ensure the Euclidean norm of new_params2 is 1
        new_s_hat /= np.linalg.norm(new_s_hat,axis = 1)[:,None]
        new_lam_c /= np.linalg.norm(new_lam_c,axis = 0)[:,None]
        new_cost = evaluate_fitness_controlled(k,K,n_f,lam,scale*new_lam_c,new_s_hat,m,t_f,g)

        if new_cost < current_cost:
            current_s_hat = new_s_hat
            current_lam_c = new_lam_c
            current_cost = new_cost

            if new_cost < best_cost:
                best_s_hat = new_s_hat
                best_lam_c = new_lam_c
                best_cost = new_cost
        else:
            probability = np.exp((current_cost - new_cost) / temperature)
            if np.random.random() < probability:
                current_s_hat = new_s_hat
                current_lam_c = new_lam_c
                current_cost = new_cost

        temperature *= cooling_rate
        #if iteration/100 ==0 :
            #print(current_cost)
    return best_s_hat, best_lam_c, costs