import numpy as np
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from copy import deepcopy
import scipy.sparse as sp
import matplotlib.pyplot as plt

##################################################################################################
# bring in Will's dictionary

DictIC = {
    # Parameters shared with the perfect foresight model
    "CRRA": 2,  # Coefficient of relative risk aversion
    "Rfree": .04**0.25,  # Interest factor on assets
    "DiscFac": 0.975,  # Intertemporal discount factor
    "LivPrb": 0.99375,  # Survival probability
    "PermGroFac": 1.0,  # Permanent income growth factor
    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd": [0.06],  # Standard deviation of log permanent shocks to income
    "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.2],  # Standard deviation of log transitory shocks to income
    "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.00,  # Probability of unemployment while working
    "IncUnemp": 0.0,  # Unemployment benefits replacement rate
    "UnempPrbRet": 0.0000,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    # A few other parameters
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    # Parameters only used in simulation
    "AgentCount": 20000,  # Number of agents of this type
    "T_sim": 1000,  # Number of periods to simulate
    "aNrmInitMean": np.log(0.000001),  # Mean of log initial assets ,
    # The value of np.log(0.0) causes the code to ensure newborns have have exactly 1.0 in market resources.
    # Mean of log initial assets
    "aNrmInitStd": 0.0,  # Standard deviation of log initial assets
    "pLvlInitMean": 0.0,  # Mean of log initial permanent income
    "pLvlInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": None,  # Age after which simulated agents are automatically killed
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin": 0.0001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 100000,  # Maximum end-of-period "assets above minimum" value
    "aXtraCount": 130,  # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac": 3,  # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra": None,  # Additional values to add to aXtraGrid
    # Parameters for Transition Matrix Simulation
    "mCount": 200,
    "mFac": 3,
    "mMin": 1e-4,
    "mMax": 10000,
    "MrkvArray": None,
    "MrkvPrbsInit": None,
}

DictMrkv = deepcopy(DictIC)
DictMrkv["Rfree"] = np.array([1.04**0.25, 1.04**0.25, 1.04**0.25, 1.04**0.25])
DictMrkv["LivPrb"] = [np.array([0.99375, 0.99375, 0.99375, 0.99375])]
DictMrkv["PermGroFac"] = [np.array([1.0, 1.0, 1.0, 1.0])]
DictMrkv["MrkvArray"] = [np.array([[0.9, 0.1, 0.0, 0.0], \
                                   [0.9, 0.05, 0.05, 0], \
                                    [0.9, 0, 0.05, 0.05], \
                                        [0.9, 0, 0, 0.1]])]
DictMrkv["MrkvPrbsInit"] = np.array([0.99, 0.01, 0.0, 0.0])

# DictMrkv["MrkvArray"] = [np.array([[0.9, 0.1, 0.0, 0.0], \
#                                    [0.7, 0.2, 0.1, 0], \
#                                     [0.7, 0, 0.2, 0.1], \
#                                         [0.7, 0, 0, 0.3]])]
# DictMrkv["MrkvPrbsInit"] = np.array([0.99, 0.01, 0.0, 0.0])

bigT = 100
dx = 0.0001

##################################################################################################
# declare agents

agentIC = IndShockConsumerType(**DictIC)
agent1 = MarkovConsumerType(**DictMrkv)
agent1.cycles = 0

##################################################################################################
# Income distributions

# HAF distributions
IncShkDstn_emp = deepcopy(agentIC.IncShkDstn[0])
IncShkDstn_emp_dx = deepcopy(IncShkDstn_emp)
IncShkDstn_emp_dx.atoms[1] = IncShkDstn_emp_dx.atoms[1] * (1 + dx)    

# quasi HAF unemp
quasiHAFue = deepcopy(IncShkDstn_emp)
quasiHAFue.atoms[0] = quasiHAFue.atoms[0] * 0 + 1.0
quasiHAFue.atoms[1] = quasiHAFue.atoms[1] * 0 + 0.7

quasiHAFuenb = deepcopy(IncShkDstn_emp)
quasiHAFuenb.atoms[0] = quasiHAFue.atoms[0] * 0 + 1.0
quasiHAFuenb.atoms[1] = quasiHAFue.atoms[1] * 0 + 0.5

IncShkDstn = [[IncShkDstn_emp, quasiHAFue, quasiHAFue, quasiHAFuenb]]
IncShkDstn_dx = [[IncShkDstn_emp_dx, quasiHAFue, quasiHAFue, quasiHAFuenb]]

##################################################################################################
# Transition Matrices

agentTM = deepcopy(agent1)
agentTM.IncShkDstn = deepcopy(IncShkDstn)

agentTM.compute_steady_state()
print("TM results:")
print(agentTM.A_ss)
print(agentTM.C_ss)
print("")

c = agentTM.cPol_Grid
a = agentTM.aPol_Grid

agentTM2 = deepcopy(agent1)
agentTM2.IncShkDstn = deepcopy(IncShkDstn_dx)
agentTM2.neutral_measure = True
agentTM2.harmenberg_income_process()

##################################################################################################
# Finite Horizon

params = deepcopy(DictMrkv)
params["T_cycle"] = bigT
params["LivPrb"] = params["T_cycle"] * [agentTM.LivPrb[0]]
params["PermGroFac"] = params["T_cycle"] * [agentTM.PermGroFac[0]]
params["PermShkStd"] = params["T_cycle"] * [agentTM.PermShkStd[0]]
params["TranShkStd"] = params["T_cycle"] * [agentTM.TranShkStd[0]]
params["Rfree"] = params["T_cycle"] * [agentTM.Rfree]
params["MrkvArray"] = params["T_cycle"] * agentTM.MrkvArray

FinHorizonAgent = MarkovConsumerType(**params)
FinHorizonAgent.cycles = 1

FinHorizonAgent.del_from_time_inv(
    "IncShkDstn",
)
FinHorizonAgent.add_to_time_vary("IncShkDstn")

FinHorizonAgent.solution_terminal = deepcopy(agentTM.solution[0])
FinHorizonAgent.IncShkDstn = (params["T_cycle"] - 1) * deepcopy(IncShkDstn) + deepcopy(IncShkDstn_dx) + deepcopy(IncShkDstn)
FinHorizonAgent.dist_pGrid = params["T_cycle"] * [np.array([1])]
FinHorizonAgent.solution_terminal = deepcopy(agentTM.solution[0])

FinHorizonAgent.solve()

# Calculate Transition Matrices
FinHorizonAgent.neutral_measure = True
# FinHorizonAgent.harmenberg_income_process()
FinHorizonAgent.IncShkDstn = (params["T_cycle"] - 1) * deepcopy(agentTM.IncShkDstn) + deepcopy(agentTM2.IncShkDstn) + deepcopy(IncShkDstn)
FinHorizonAgent.define_distribution_grid()
FinHorizonAgent.calc_transition_matrix() 

##################################################################################################
# period zero shock agent

Zeroth_col_agent = MarkovConsumerType(**params)
Zeroth_col_agent.cycles = 1 
Zeroth_col_agent.solution_terminal = deepcopy(agentTM.solution[0])
Zeroth_col_agent.IncShkDstn = params["T_cycle"] * deepcopy(IncShkDstn) 
Zeroth_col_agent.solve()
Zeroth_col_agent.IncShkDstn = deepcopy(agentTM2.IncShkDstn) + (params["T_cycle"]) * deepcopy(agentTM.IncShkDstn)
Zeroth_col_agent.neutral_measure = True
# Zeroth_col_agent.harmenberg_income_process()
Zeroth_col_agent.define_distribution_grid()
Zeroth_col_agent.calc_transition_matrix()

##################################################################################################
# calculate Jacobian

D_ss = agentTM.vec_erg_dstn

c_ss = agentTM.cPol_Grid.flatten()
a_ss = agentTM.aPol_Grid.flatten()

c_t_unflat = FinHorizonAgent.cPol_Grid
a_t_unflat = FinHorizonAgent.aPol_Grid

A_ss = agentTM.A_ss
C_ss = agentTM.C_ss
    
transition_matrices = FinHorizonAgent.tran_matrix

c_t_flat = np.zeros((params["T_cycle"], int(params["mCount"] * 4)))
a_t_flat = np.zeros((params["T_cycle"], int(params["mCount"] * 4)))

# c_t_flat = np.zeros((params["T_cycle"], params["mCount"], 4))
# a_t_flat = np.zeros((params["T_cycle"], params["mCount"], 4))

for t in range( params["T_cycle"] ):
    c_t_flat[t] = c_t_unflat[t].flatten()
    a_t_flat[t] = a_t_unflat[t].flatten()

tranmat_ss = agentTM.tran_matrix

tranmat_t = np.insert(transition_matrices, params["T_cycle"], tranmat_ss, axis = 0)

c_t = np.insert(c_t_flat, params["T_cycle"] , c_ss , axis = 0)
a_t = np.insert(a_t_flat, params["T_cycle"] , a_ss , axis = 0)



##################################################################################################

def compile_JAC(a_ss, c_ss, a_t, c_t, tranmat_ss, tranmat_t, D_ss, C_ss, A_ss):

    T = params['T_cycle']

    # Expectation vectors
    exp_vecs_a_e = []
    exp_vec_a_e = a_ss
    
    exp_vecs_c_e = []
    exp_vec_c_e = c_ss
    
    for i in range(T):
        
        exp_vecs_a_e.append(exp_vec_a_e)
        exp_vec_a_e = np.dot(tranmat_ss.T, exp_vec_a_e)
        
        exp_vecs_c_e.append(exp_vec_c_e)
        exp_vec_c_e = np.dot(tranmat_ss.T, exp_vec_c_e)
    
    
    exp_vecs_a_e = np.array(exp_vecs_a_e)
    exp_vecs_c_e = np.array(exp_vecs_c_e)

    
    da0_s = []
    dc0_s = []

    for i in range(T):
        da0_s.append(a_t[T - i] - a_ss)
        dc0_s.append(c_t[T - i] - c_ss)
    
        
    da0_s = np.array(da0_s)
    dc0_s = np.array(dc0_s)

    dA0_s = []
    dC0_s = []

    for i in range(T):
        dA0_s.append(np.dot(da0_s[i], D_ss))
        dC0_s.append(np.dot(dc0_s[i], D_ss))
    
    dA0_s = np.array(dA0_s)
    A_curl_s = dA0_s/dx
    
    dC0_s = np.array(dC0_s)
    C_curl_s = dC0_s/dx
    
    dlambda0_s = []
    
    for i in range(T):
        dlambda0_s.append(tranmat_t[T - i] - tranmat_ss)
    
    dlambda0_s = np.array(dlambda0_s)
    
    dD0_s = []
    
    for i in range(T):
        dD0_s.append(np.dot(dlambda0_s[i], D_ss))
    
    dD0_s = np.array(dD0_s)
    D_curl_s = dD0_s/dx
    
    Curl_F_A = np.zeros((T , T))
    Curl_F_C = np.zeros((T , T))
    
    # WARNING: SWAPPED THESE LINES TO MAKE DEMO RUN
    # Curl_F_A[0] = A_curl_s
    # Curl_F_C[0] = C_curl_s
    Curl_F_A[0] = A_curl_s.T[0]
    Curl_F_C[0] = C_curl_s.T[0]

    for i in range(T-1):
        for j in range(T):
            Curl_F_A[i + 1][j] = np.dot(exp_vecs_a_e[i], D_curl_s[j])
            Curl_F_C[i + 1][j] = np.dot(exp_vecs_c_e[i], D_curl_s[j])

    J_A = np.zeros((T, T))
    J_C = np.zeros((T, T))

    for t in range(T):
        for s in range(T):
            if (t ==0) or (s==0):
                J_A[t][s] = Curl_F_A[t][s]
                J_C[t][s] = Curl_F_C[t][s]
                
            else:
                J_A[t][s] = J_A[t - 1][s - 1] + Curl_F_A[t][s]
                J_C[t][s] = J_C[t - 1][s - 1] + Curl_F_C[t][s]
     
    # Zeroth Column of the Jacobian
    Zeroth_col_agent.tran_matrix = np.array(Zeroth_col_agent.tran_matrix)
    
    C_t = np.zeros(T)
    A_t = np.zeros(T)
    
    dstn_dot = D_ss
    
    for t in range(T):
        tran_mat_t = Zeroth_col_agent.tran_matrix[t]

        dstn_all = np.dot(tran_mat_t, dstn_dot)

        C = np.dot(c_ss, dstn_all)
        A = np.dot(a_ss, dstn_all)
        
        C_t[t] = C
        A_t[t] = A

        dstn_dot = dstn_all
        
    J_A.T[0] = (A_t - A_ss) / dx
    J_C.T[0] = (C_t - C_ss) / dx

    return J_C, J_A


CJAC_perfect, AJAC_perfect = compile_JAC(a_ss, c_ss, a_t, c_t, tranmat_ss, tranmat_t, D_ss, C_ss, A_ss)

plt.plot(CJAC_perfect.T[0])
plt.plot(CJAC_perfect.T[10])
plt.plot(CJAC_perfect.T[30])
plt.plot(CJAC_perfect.T[50])

plt.plot(np.zeros(100) , 'k')
plt.title('Consumption Jacobian')
plt.show()

plt.plot(AJAC_perfect.T[0])
plt.plot(AJAC_perfect.T[10])
plt.plot(AJAC_perfect.T[30])
plt.plot(AJAC_perfect.T[50])

plt.plot(np.zeros(100) , 'k')
plt.title('Asset Jacobian')
plt.show()

##################################################################################################
