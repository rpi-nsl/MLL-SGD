import numpy as np
import matplotlib.pyplot as plt
import pickle
font = {'family' : 'DejaVu Sans',
#        'weight' : 'bold',
        'size'   : 16}
plt.rc('font', **font)

models = [1,3,0]
datasets = [1,2,0]

def stagger_loss(costs, tau, q):
    costs_new = np.zeros(epochs)
    filename = f"results/coin_model{model_type}_data{dataset}_hubs10_workers10_tau{tau}_q{q}_graph{graph}_prob5.p"
    coins_old = pickle.load(open(filename,'rb'))
    coins = np.zeros((epochs*q*tau,10,10))
    for h in range(10):
        for c in range(10):
            for e in range(epochs*q):
                for t in range(tau):
                    if coins_old[e,h,c] > 0:
                        coins[e*tau+t,h,c] = 1
                        coins_old[e,h,c] -= 1
        
    inds = np.zeros((10,10))
    client_done = np.zeros((10,10))
    client_done.fill(False)
    client_epochs = np.zeros((10,10))
    hub_done = np.zeros(10)
    hub_inds = np.zeros(10)
    hub_done.fill(False)
    # For all epochs
    last_loss = costs[0]
    old_e = 0
    for e in range(epochs):
        costs_new[e] = last_loss
        for i in range(q*tau):
            for h in range(10):
                if hub_done[h]: 
                    continue
                for c in range(10):
                    if client_done[h,c]: 
                        continue

                    if coins[int(client_epochs[h,c]),h,c] == 1:
                        inds[h,c] += 1

                    # Check if has done tau steps 
                    if inds[h,c] >= tau:
                        inds[h,c] = 0 
                        client_done[h,c] = True

                    client_epochs[h,c] += 1

                # Check if all clients have finished
                if np.all(client_done[h]):
                    hub_inds[h] += 1
                    if hub_inds[h] >= q:
                        hub_inds[h] = 0
                        hub_done[h] = True
                    else:
                        for c in range(10):
                            client_done[h,c] = False 

            # Check if all hubs have finished
            if np.all(hub_done):
                old_e += 1
                last_loss = costs[old_e]
                for h in range(10):
                    hub_done[h] = False 
                    for c in range(10):
                        client_done[h,c] = False 
    return costs_new

for m in range(3):
    epochs = 1000
    if m < 2:
        epochs = 200
    timesteps = range(0,epochs*32,32)
    model_type = models[m]
    dataset = datasets[m]
    types = ['loss', 'accuracy']

    ###################### Experiment 1 ############################
    for t in types:
        fig, ax = plt.subplots()
        # Distributed SGD
        filename = f"results/{t}_model{model_type}_data{dataset}_hubs1_workers100_tau1_q32_graph5_prob0fed.p"
        costs = pickle.load(open(filename,'rb'))
        plt.plot(timesteps, costs[:epochs], label="Distributed SGD")

        # Run Local SGD
        filename = f"results/{t}_model{model_type}_data{dataset}_hubs1_workers100_tau32_q1_graph5_prob0fed.p"
        costs = pickle.load(open(filename,'rb'))
        plt.plot(timesteps, costs[:epochs], label="Local SGD \u03C4=32")

        # Run MLL-SGD variations
        hubs = 10
        workers = 10
        graph = 5
        prob = 0
        q = [4,8]
        tau = [8,4]
        for i in range(2):
            filename = f"results/{t}_model{model_type}_data{dataset}_hubs{hubs}_workers{workers}_tau{tau[i]}_q{q[i]}_graph{graph}_prob{prob}fed.p"
            costs = pickle.load(open(filename,'rb'))
            plt.plot(timesteps, costs[:epochs], label="MLL-SGD q="+str(q[i])+" \u03C4="+str(tau[i]))

        if t == 'loss':
            plt.legend(loc="upper right")
            plt.ylabel('Loss');
            if m == 0:
                plt.ylim(0,1)
            if m == 1:
                plt.ylim(-0.1,2)
            if m == 2:
                plt.ylim(0.32, 0.375)
        else:
            plt.legend(loc="lower right")
            plt.ylabel('Accuracy');
            if m == 0:
                plt.ylim(0.6,1)
            if m == 1:
                plt.ylim(0.3,0.5)
        plt.xlabel('k');
        plt.tight_layout()

        # Set aspect ratio
        ratio = 0.5
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

        plt.savefig("images/exp1new"+str(m)+"_"+t+".png", dpi=300, aspect='preserve')


    ####################### Experiment 2 ############################
    fig, ax = plt.subplots()
    names = ['Prob=1', 'Fixed', 'Uniform Dist', 'Skewed 1', 'Skewed 2']
    hubs = 10
    workers = 10
    tau = 8
    q = 4

    ## Distributed SGD
    #filename = f"results/loss_model{model_type}_data{dataset}_hubs1_workers100_tau1_q32_graph5_prob0.p"
    #costs = pickle.load(open(filename,'rb'))
    #plt.plot(timesteps, costs, label="Distributed SGD")

    # Run MLL-SGD variations
    graph = 3
    for i in range(5):
        filename = f"results/loss_model{model_type}_data{dataset}_hubs{hubs}_workers{workers}_tau{tau}_q{q}_graph{graph}_prob{i}.p"
        costs = pickle.load(open(filename,'rb'))
        plt.plot(timesteps, costs[:epochs], label=names[i])

    plt.legend(loc="upper right")
    plt.xlabel('k');
    plt.ylabel('Loss');
    if m == 0:
        plt.ylim(0.1,1)
    if m == 1:
        plt.ylim(-0.1,0.5)
    if m == 2:
        plt.ylim(0.32, 0.375)
    plt.tight_layout()

    # Set aspect ratio
    ratio = 0.5
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

    plt.savefig("images/exp2new"+str(m)+".png", dpi=300)

    ###################### Experiment 3 ###########################
    fig, ax = plt.subplots()
    names = ['Hubs=5 Workers=20',
            'Hubs=10 Workers=10',
            'Hubs=20 Workers=5']
    tau = 8
    q = 4
    g = 6
    fig, ax = plt.subplots()
    # Distributed SGD
    filename = f"results/loss_model{model_type}_data{dataset}_hubs1_workers100_tau1_q32_graph5_prob0.p"
    costs = pickle.load(open(filename,'rb'))
    plt.plot(timesteps, costs[:epochs], label="Distributed SGD")
    # Run Local SGD
    filename = f"results/loss_model{model_type}_data{dataset}_hubs10_workers10_tau32_q1_graph5_prob0.p"
    costs = pickle.load(open(filename,'rb'))
    plt.plot(timesteps, costs[:epochs], label="Local SGD \u03C4=32")
    # Run MLL-SGD variations
    hubs = [5, 10, 20]
    workers = [20, 10, 5]
    for i in range(3):
        filename = f"results/loss_model{model_type}_data{dataset}_hubs{hubs[i]}_workers{workers[i]}_tau{tau}_q{q}_graph{g}_prob0.p"
        costs = pickle.load(open(filename,'rb'))
        plt.plot(timesteps, costs[:epochs], label=names[i])

    plt.legend(loc="upper right")
    plt.xlabel('k');
    plt.ylabel('Loss');
    if m == 0:
        plt.ylim(0.1,1)
    if m == 1:
        plt.ylim(-0.1,2)
    if m == 2:
        plt.ylim(0.32, 0.375)
    plt.tight_layout()

    # Set aspect ratio
    ratio = 0.5
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

    plt.savefig("images/exp3new"+str(m)+".png", dpi=300)


    ############################ Experiment 4 ############################
    for t in types:
        fig, ax = plt.subplots()
        graph = 5
        chances = [0.2,0.4,0.6,0.8]
        i = 0
        fig, ax = plt.subplots()
        # MLL-SGD
        tau = 32
        q = 1
        filename = f"results/{t}_model{model_type}_data{dataset}_hubs10_workers10_tau{tau}_q{q}_graph{graph}_prob5.p"
        costs = pickle.load(open(filename,'rb'))
        plt.plot(timesteps, costs[:epochs], label="MLL-SGD q="+str(q)+" \u03C4="+str(tau))

        # Local SGD with probability offset
        filename = f"results/{t}_model{model_type}_data{dataset}_hubs10_workers10_tau32_q1_graph{graph}_prob0.p"
        costs = pickle.load(open(filename,'rb'))
        costs_new = stagger_loss(costs,32,1)
        plt.plot(timesteps, costs_new[:epochs], label="Local SGD \u03C4=32")

        # MLL-SGD
        tau = 8
        q = 4
        filename = f"results/{t}_model{model_type}_data{dataset}_hubs10_workers10_tau{tau}_q{q}_graph{graph}_prob5.p"
        costs = pickle.load(open(filename,'rb'))
        plt.plot(timesteps, costs[:epochs], label="MLL-SGD q="+str(q)+" \u03C4="+str(tau))

        # HL-SGD with probability offset
        filename = f"results/{t}_model{model_type}_data{dataset}_hubs10_workers10_tau8_q4_graph{graph}_prob0.p"
        costs = pickle.load(open(filename,'rb'))
        costs_new = stagger_loss(costs,8,4)
        plt.plot(timesteps, costs_new[:epochs], label="HL-SGD q="+str(q)+" \u03C4=8")

        if t == 'loss':
            plt.legend(loc="upper right")
            plt.ylabel('Loss');
            if m == 0:
                plt.ylim(0.1,1)
            if m == 1:
                plt.ylim(-0.1,2)
            if m == 2:
                plt.ylim(0.32, 0.375)
        else:
            plt.legend(loc="lower right")
            plt.ylabel('Accuracy');
            if m == 0:
                plt.ylim(0.6,1)
            if m == 1:
                plt.ylim(0.3,0.475)
        plt.xlabel('Time slots');
        plt.tight_layout()

        # Set aspect ratio
        ratio = 0.5
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

        plt.savefig("images/exp4new"+str(m)+"_"+t+".png", dpi=300)
