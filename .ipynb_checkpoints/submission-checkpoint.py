import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏󠄋͏︇͏󠄋͏󠄐
import random # random for sampling probs
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏󠄋͏︇͏󠄋͏󠄐
#
# pgmpy.sampling.*͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏󠄋͏︇͏󠄋͏󠄐
# pgmpy.factors.*͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏󠄋͏︇͏󠄋͏󠄐
# pgmpy.estimators.*͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏󠄋͏︇͏󠄋͏󠄐

def make_security_system_net():
    """
        Create a Bayes Net representation of the above security system problem. 
        Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
        "D"'. (for the tests to work.)
    """
    BayesNet = BayesianNetwork()
    
    BayesNet.add_node("H")
    BayesNet.add_node("C")
    BayesNet.add_node("Q")
    BayesNet.add_node("D")
    BayesNet.add_node("B")
    BayesNet.add_node("M")
    BayesNet.add_node("K")

    BayesNet.add_edge("H","Q")
    BayesNet.add_edge("C","Q")
    BayesNet.add_edge("Q","D")
    BayesNet.add_edge("M","K")
    BayesNet.add_edge("B","K")
    BayesNet.add_edge("K","D")
    return BayesNet

def set_probability(bayes_net):
    """
        Set probability distribution for each node in the security system.
        Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
        "D"'. (for the tests to work.)
        Probabilities:

        P(H)= F: .5, T: .5
        P(C)= F: .7, T: .3
        P(M)= F: .2, T: .8
        P(B)= F: .5, T: .5
        
        P(Q|C’) = F: .45, T: .55
        P(Q|C) = F: .1, T: .9
        
        P(Q’| C, H’)= F: .25, T: .75 (No hack on Q given Contra and no professional Hackers)
        P(Q| C, H’)= F: .75, T: .25
        
        P(Q’| C’, H’)= F: .05, T: .95 (No hack on Q given no Contra and no professional Hackers)
        P(Q| C’, H’)= F: .95, T: .05
        
        P(K|M, B) = F .15, T .85 (Gets Kidnapped given Mercenary and Bond Protection)
        P(K|M’, B) = F .01, T .99 (Gets Kidnapped given no Mercenary and Bond Protection)
        
        P(K|B’, M) = F .05, T .95 (Kidnapped given no Bond and Mercenary attack)
        P(K|B’, M’) = F .25 T .75 (Kidnapped given no Bond and no Mercenary attack)
        
        P(D|Q’, K’) = F .01 T .99 (Double-0 files obtained )
        P(D|Q, K) = F .98 T .02
        P(D|Q’, K’) = F .4 T .6
        P(D|K, Q’) = F .65 T .35

    """
    cpd_h = TabularCPD('H', 2, values=[[0.5], [0.5]])
    cpd_c = TabularCPD('C', 2, values=[[0.7], [0.3]])
    cpd_m = TabularCPD('M', 2, values=[[0.2], [0.8]]) 
    cpd_b = TabularCPD('B', 2, values=[[0.5], [0.5]])
    
    cpd_qhc = TabularCPD('Q', 2, values=[[0.95, 0.75, 0.45, 0.1], \
                    [0.05, 0.25, 0.55, 0.9]], evidence=['H', 'C'], evidence_card=[2, 2])

    cpd_kbm = TabularCPD('K', 2, values=[[0.25, 0.99, 0.05, 0.85], \
                    [0.75, 0.01, 0.95, 0.15]], evidence=['M', 'B'], evidence_card=[2, 2])

    cpd_dqk = TabularCPD('D', 2, values=[[0.98, 0.65, 0.4, 0.01], \
                    [0.02, 0.35, 0.6, 0.99]], evidence=['Q', 'K'], evidence_card=[2, 2])

    #print(cpd_h.get_values())
    #print(cpd_c.get_values())
    #print(cpd_m.get_values())
    #print(cpd_b.get_values())
    #print(cpd_qhc.get_values())
    #print(cpd_kbm.get_values())
    #print(cpd_dqk.get_values())
    
    bayes_net.add_cpds(cpd_h, cpd_c, cpd_m, cpd_b, cpd_qhc, cpd_kbm, cpd_dqk)
    return bayes_net

def get_marginal_double0(bayes_net):
    """
        Calculate the marginal probability that Double-0 gets compromised.
    """    
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['D'], joint=False)
    double0_prob = marginal_prob['D'].values
    #print(double0_prob[1])
    return double0_prob[1]

def get_conditional_double0_given_no_contra(bayes_net):
    """
        Calculate the conditional probability that Double-0 gets compromised
        given Contra is shut down.
    """
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'], evidence={'C': 0}, joint=False)
    double0_prob = conditional_prob['D'].values
    return double0_prob[1]

def get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net):
    """
        Calculate the conditional probability that Double-0 gets compromised
        given Contra is shut down and Bond is reassigned to protect M.
    """
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'], evidence={'C': 0, 'B': 1}, joint=False)
    double0_prob = conditional_prob['D'].values
    return double0_prob[1]

def get_game_network():
    """
        Create a Bayes Net representation of the game problem.
        Name the nodes as "A","B","C","AvB","BvC" and "CvA".  
    """
    BayesNet = BayesianNetwork()
    BayesNet.add_nodes_from(['A', 'B', 'C', 'AvB', 'BvC', 'CvA'])
    BayesNet.add_edges_from([('A', 'AvB'), ('B', 'AvB'),
                             ('B', 'BvC'), ('C', 'BvC'),
                             ('C', 'CvA'), ('A', 'CvA')])

    cpd_A = TabularCPD(variable='A', variable_card=4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_B = TabularCPD(variable='B', variable_card=4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_C = TabularCPD(variable='C', variable_card=4, values=[[0.15], [0.45], [0.30], [0.10]])

    def get_match_probabilities(diff):
        diff_abs = abs(diff)
        if diff_abs == 0:
            probs = [0.10, 0.10, 0.80]
        elif diff_abs == 1:
            probs = [0.20, 0.60, 0.20]
        elif diff_abs == 2:
            probs = [0.15, 0.75, 0.10]
        elif diff_abs == 3:
            probs = [0.05, 0.90, 0.05]
        else:
            raise ValueError("Invalid skill difference")

        if diff >= 0:
            return probs
        else:
            return [probs[1], probs[0], probs[2]]

    match_cpd_values = []
    for i in range(4): 
        for j in range(4): 
            diff = j - i
            probs = get_match_probabilities(diff)
            match_cpd_values.append(probs)

    match_cpd_values_T = list(zip(*match_cpd_values))

    cpd_AvB = TabularCPD(variable='AvB', variable_card=3,
                         values=match_cpd_values_T,
                         evidence=['A', 'B'], evidence_card=[4, 4])

    cpd_BvC = TabularCPD(variable='BvC', variable_card=3,
                         values=match_cpd_values_T,
                         evidence=['B', 'C'], evidence_card=[4, 4])

    cpd_CvA = TabularCPD(variable='CvA', variable_card=3,
                         values=match_cpd_values_T,
                         evidence=['C', 'A'], evidence_card=[4, 4])

    BayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_AvB, cpd_BvC, cpd_CvA)
    return BayesNet

def calculate_posterior(bayes_net):
    """
        Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
        Return a list of probabilities corresponding to win, loss and tie likelihood.
    """
    posterior = [0,0,0]
    
    solver = VariableElimination(bayes_net)
    evidence = {'AvB': 0, 'CvA': 2}
    posterior_distribution = solver.query(variables=['BvC'], evidence=evidence)
    posterior = posterior_distribution.values.tolist()
    return posterior 

def Gibbs_sampler(bayes_net, initial_state):
    """
        Complete a single iteration of the Gibbs sampling algorithm 
        given a Bayesian network and an initial state value. 
        
        initial_state is a list of length 6 where: 
        index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
        index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
        
        Returns the new state sampled from the probability distribution as a tuple of length 6.
        Return the sample as a tuple. 

        Note: You are allowed to calculate the probabilities for each potential variable
        to be sampled. See README for suggested structure of the sampling process.
    """
    # Handle initial_state being None or empty
    if not initial_state or len(initial_state) != 6:
        initial_state = [
            random.randint(0, 3),  
            random.randint(0, 3),  
            random.randint(0, 3),  
            0,                    
            random.randint(0, 2),  
            2                      
        ]
    
    sample = tuple(initial_state)    
    non_evidence_indices = [0,1,2,4]
    variable_index = random.choice(non_evidence_indices)
    state = list(sample)
    if variable_index == 0:
        A_values = [0,1,2,3]
        probs = []
        for A_val in A_values:
            A_cpd = bayes_net.get_cpds('A')
            P_A = A_cpd.values[A_val]
            AvB_cpd = bayes_net.get_cpds('AvB')
            P_AvB = AvB_cpd.values[state[3], A_val, state[1]]
            CvA_cpd = bayes_net.get_cpds('CvA')
            P_CvA = CvA_cpd.values[state[5], state[2], A_val]
            prob = P_A * P_AvB * P_CvA
            probs.append(prob)
        total = sum(probs)
        normalized_probs = [p/total for p in probs]
        new_A = random.choices(A_values, weights=normalized_probs)[0]
        state[0] = new_A
    elif variable_index == 1:
        B_values = [0,1,2,3]
        probs = []
        for B_val in B_values:
            B_cpd = bayes_net.get_cpds('B')
            P_B = B_cpd.values[B_val]
            AvB_cpd = bayes_net.get_cpds('AvB')
            P_AvB = AvB_cpd.values[state[3], state[0], B_val]
            BvC_cpd = bayes_net.get_cpds('BvC')
            P_BvC = BvC_cpd.values[state[4], B_val, state[2]]
            prob = P_B * P_AvB * P_BvC
            probs.append(prob)
        total = sum(probs)
        normalized_probs = [p/total for p in probs]
        new_B = random.choices(B_values, weights=normalized_probs)[0]
        state[1] = new_B
    elif variable_index == 2:
        C_values = [0,1,2,3]
        probs = []
        for C_val in C_values:
            C_cpd = bayes_net.get_cpds('C')
            P_C = C_cpd.values[C_val]
            BvC_cpd = bayes_net.get_cpds('BvC')
            P_BvC = BvC_cpd.values[state[4], state[1], C_val]
            CvA_cpd = bayes_net.get_cpds('CvA')
            P_CvA = CvA_cpd.values[state[5], C_val, state[0]]
            prob = P_C * P_BvC * P_CvA
            probs.append(prob)
        total = sum(probs)
        normalized_probs = [p/total for p in probs]
        new_C = random.choices(C_values, weights=normalized_probs)[0]
        state[2] = new_C
    elif variable_index == 4:
        BvC_values = [0,1,2]
        probs = []
        BvC_cpd = bayes_net.get_cpds('BvC')
        for BvC_val in BvC_values:
            P_BvC = BvC_cpd.values[BvC_val, state[1], state[2]]
            probs.append(P_BvC)
        total = sum(probs)
        normalized_probs = [p/total for p in probs]
        new_BvC = random.choices(BvC_values, weights=normalized_probs)[0]
        state[4] = new_BvC
    else:
        raise ValueError("Variable index out of range")
    sample = tuple(state)
    return sample


def MH_sampler(bayes_net, initial_state):
    """
        Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
        initial_state is a list of length 6 where: 
        index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
        index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
        Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """

    if not initial_state or len(initial_state) != 6:
        initial_state = [
            random.randint(0, 3),  
            random.randint(0, 3),  
            random.randint(0, 3), 
            0,                     
            random.randint(0, 2),  
            2                      
        ]

    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    sample = tuple(initial_state)    
    state = list(sample)
    candidate_state = state.copy()
    non_evidence_indices = [0,1,2,4]
    for idx in non_evidence_indices:
        if idx in [0,1,2]:
            candidate_state[idx] = random.randint(0,3)
        elif idx == 4:
            candidate_state[idx] = random.randint(0,2)
    def compute_joint_prob(state):
        A_val = state[0]
        B_val = state[1]
        C_val = state[2]
        AvB_val = state[3]
        BvC_val = state[4]
        CvA_val = state[5]
        P_A = team_table[A_val]
        B_cpd = bayes_net.get_cpds('B')
        P_B = B_cpd.values[B_val]
        C_cpd = bayes_net.get_cpds('C')
        P_C = C_cpd.values[C_val]
        P_AvB = match_table[AvB_val, A_val, B_val]
        BvC_cpd = bayes_net.get_cpds('BvC')
        P_BvC = BvC_cpd.values[BvC_val, B_val, C_val]
        CvA_cpd = bayes_net.get_cpds('CvA')
        P_CvA = CvA_cpd.values[CvA_val, C_val, A_val]
        joint_prob = P_A * P_B * P_C * P_AvB * P_BvC * P_CvA
        return joint_prob
    P_current = compute_joint_prob(state)
    P_candidate = compute_joint_prob(candidate_state)
    alpha = min(1, P_candidate / P_current) if P_current > 0 else 1
    if random.random() < alpha:
        sample = tuple(candidate_state)
    else:
        sample = tuple(state)    
    return sample
    

def compare_sampling(bayes_net, initial_state):
    """
        Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge.
    """    

    if not initial_state or len(initial_state) != 6:
        initial_state = [
            random.randint(0, 3),  
            random.randint(0, 3),  
            random.randint(0, 3), 
            0,                     
            random.randint(0, 2),  
            2                      
        ]
    
    initial_state = list(initial_state)
    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH

    delta = 0.001
    N = 10
    Gibbs_state = initial_state.copy()
    Gibbs_counts = [0, 0, 0]
    Gibbs_probs_history = []
    converged = False
    while not converged:
        Gibbs_count += 1
        Gibbs_state = Gibbs_sampler(bayes_net, Gibbs_state)
        BvC_outcome = Gibbs_state[4]
        Gibbs_counts[BvC_outcome] += 1
        total = sum(Gibbs_counts)
        Gibbs_probs = [count / total for count in Gibbs_counts]
        Gibbs_probs_history.append(Gibbs_probs)
        if len(Gibbs_probs_history) >= N + 1:
            converged = True
            for i in range(1, N + 1):
                current_probs = Gibbs_probs_history[-i]
                previous_probs = Gibbs_probs_history[-i - 1]
                diffs = [abs(current_probs[j] - previous_probs[j]) for j in range(3)]
                max_diff = max(diffs)
                if max_diff >= delta:
                    converged = False
                    break
    Gibbs_convergence = Gibbs_probs

    MH_state = initial_state.copy()
    MH_counts = [0, 0, 0]
    MH_probs_history = []
    converged = False
    while not converged:
        MH_count += 1
        prev_state = MH_state 
        MH_state = MH_sampler(bayes_net, MH_state)
        if MH_state == prev_state:
            MH_rejection_count += 1
        BvC_outcome = MH_state[4]
        MH_counts[BvC_outcome] += 1
        total = sum(MH_counts)
        MH_probs = [count / total for count in MH_counts]
        MH_probs_history.append(MH_probs)
        if len(MH_probs_history) >= N + 1:
            converged = True
            for i in range(1, N + 1):
                current_probs = MH_probs_history[-i]
                previous_probs = MH_probs_history[-i - 1]
                diffs = [abs(current_probs[j] - previous_probs[j]) for j in range(3)]
                max_diff = max(diffs)
                if max_diff >= delta:
                    converged = False
                    break
    MH_convergence = MH_probs       
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """
        Question about sampling performance.
    """

    choice = 0
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1.5
    return options[choice], factor


def return_your_name():
    """
        Return your name from this function
    """
    return "Sehajpreet Gandhi"
