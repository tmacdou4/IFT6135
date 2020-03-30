import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(20) + 1
exp = {}
exp[1] = []
exp[2] = []
exp[3] = []
exp[4] = []

# exp[1].append(np.load("learning_curves_test.npy", allow_pickle=True)[()])

exp[1].append(np.load("RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_save_best_0/learning_curves.npy", allow_pickle=True)[()])
exp[1].append(np.load("RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=20_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()])

exp[1].append(np.load("RNN_SGD_model=RNN_optimizer=SGD_initial_lr=10.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_1/learning_curves.npy", allow_pickle=True)[()])
#Adjust the 14th validation perplexity for this model.
#exp[1][2]["val_ppls"][14] = 0
#exp[1][2]["val_ppls"][13] = 0
#exp[1][2]["val_ppls"][12] = 0


exp[1].append(np.load("RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()])
exp[1].append(np.load("RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()])

exp[2].append(np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=20_save_best_0/learning_curves.npy", allow_pickle=True)[()])
exp[2].append(np.load("GRU_SGD_model=GRU_optimizer=SGD_initial_lr=10.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()])
exp[2].append(np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=20_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()])

exp[3].append(np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=256_num_layers=2_dp_keep_prob=0.2_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()])
exp[3].append(np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=2048_num_layers=2_dp_keep_prob=0.5_num_epochs=20_1/learning_curves.npy", allow_pickle=True)[()])
exp[3].append(np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=4_dp_keep_prob=0.5_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()])

exp[4].append(np.load("TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=64_seq_len=35_hidden_size=512_num_layers=6_dp_keep_prob=0.9_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()])
exp[4].append(np.load("TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=64_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.9_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()])
exp[4].append(np.load("TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=64_seq_len=35_hidden_size=2048_num_layers=2_dp_keep_prob=0.6_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()])
exp[4].append(np.load("TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=64_seq_len=35_hidden_size=1024_num_layers=6_dp_keep_prob=0.9_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()])

labels = {}
labels[1] = []
labels[2] = []
labels[3] = []
labels[4] = []

labels[1].append("Opt: SGD, LR: 1.0, Batch Size: 128")
labels[1].append("Opt: SGD, LR: 1.0, Batch Size: 20")
labels[1].append("Opt: SGD, LR: 10.0, Batch Size: 128")
labels[1].append("Opt: ADAM, LR: 0.001, Batch Size: 128")
labels[1].append("Opt: ADAM, LR: 0.0001, Batch Size: 128")

labels[2].append("Opt: ADAM, LR: 0.001, Batch Size: 128")
labels[2].append("Opt: SGD, LR: 10.0, Batch Size: 20")
labels[2].append("Opt: ADAM, LR: 0.001, Batch Size: 20")

labels[3].append("Hidden Size: 256, Num Layers: 2")
labels[3].append("Hidden Size: 2048, Num Layers: 2")
labels[3].append("Hidden Size: 512, Num Layers: 4")

labels[4].append("Hidden Size: 512, Num Layers: 6")
labels[4].append("Hidden Size: 512, Num Layers: 2")
labels[4].append("Hidden Size: 2048, Num Layers: 2")
labels[4].append("Hidden Size: 1024, Num Layers: 6")

colours = [(1,0,0), (0,1,0), (0,0,1), (0,1,1), (1,0,1)]

for id in range(1, 5):
    #epoch plot
    fig, ax = plt.subplots(1, 1)
    for p in range(len(exp[id])):
        plt.plot(epochs, exp[id][p]["train_ppls"], label=labels[id][p]+", Train", marker="o", color=colours[p])
        plt.plot(epochs, exp[id][p]["val_ppls"], label=labels[id][p]+", Valid.", marker="^", color=colours[p])    
    plt.ylabel("Model Perplexity")
    plt.xlabel("Epoch")
    if id == 1:
        plt.yscale("log")
        plt.legend(prop={'size': 6})
        plt.savefig("figures/3_" + str(id) + "_epochs_log")
    else:
        plt.legend(prop={'size': 9})
        plt.savefig("figures/3_" + str(id) + "_epochs")
    plt.show()

    #time plot
    fig, ax = plt.subplots(1, 1)
    for p in range(len(exp[id])):
        times = []
        times.append(exp[id][p]["times"][0])
        for i in range(1, len(exp[id][p]["times"])):
            times.append(exp[id][p]["times"][i] + times[i-1])
        plt.plot(times, exp[id][p]["train_ppls"], label=labels[id][p]+", Train", marker="o", color=colours[p])
        plt.plot(times, exp[id][p]["val_ppls"], label=labels[id][p]+", Valid.", marker="^", color=colours[p])
    plt.legend(prop={'size': 9})
    plt.ylabel("Model Perplexity")
    plt.xlabel("Total Elapsed Time in Seconds")
    
    if id == 1:
        plt.yscale("log")
        plt.savefig("figures/3_" + str(id) + "_time_log")
    else:
        plt.savefig("figures/3_" + str(id) + "_time")
    plt.show()
    
    if id == 1:
        #remove the colour that the removed datapoints would be, so that they match
        new_colours = [(1,0,0), (0,1,0), (0,1,1), (1,0,1)]
        del exp[1][2]
        #epoch plot
        fig, ax = plt.subplots(1, 1)
        for p in range(len(exp[id])):
            plt.plot(epochs, exp[id][p]["train_ppls"], label=labels[id][p]+", Train", marker="o", color=new_colours[p])
            plt.plot(epochs, exp[id][p]["val_ppls"], label=labels[id][p]+", Valid.", marker="^", color=new_colours[p])
        plt.legend(prop={'size': 9})    
        plt.ylabel("Model Perplexity")
        plt.xlabel("Epoch")
        plt.savefig("figures/3_" + str(id) + "_epochs_removed")
        
        #time plot
        fig, ax = plt.subplots(1, 1)
        for p in range(len(exp[id])):
            times = []
            times.append(exp[id][p]["times"][0])
            for i in range(1, len(exp[id][p]["times"])):
                times.append(exp[id][p]["times"][i] + times[i-1])
            plt.plot(times, exp[id][p]["train_ppls"], label=labels[id][p]+", Train", marker="o", color=new_colours[p])
            plt.plot(times, exp[id][p]["val_ppls"], label=labels[id][p]+", Valid.", marker="^", color=new_colours[p])
        plt.legend(prop={'size': 9})
        plt.ylabel("Model Perplexity")
        plt.xlabel("Total Elapsed Time in Seconds")
        plt.savefig("figures/3_" + str(id) + "_time_removed")
        
