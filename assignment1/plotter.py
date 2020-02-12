import matplotlib.pyplot as plt
import numpy as np

# #Figure for initialization plot
fig = plt.figure()
ax = fig.add_subplot(111)

epochs = np.arange(1,11)

glorot = [1.940678236481538, 1.5320243588474867, 1.1881688857625758, 0.9509566556778705, 0.796326276306398, 0.6931760961900418, 0.6212387494994682, 0.568835219342895, 0.5291699708452022, 0.4981390800380413]
normal = [2.3662454197929503, 1.8342111393373057, 1.5627477018530187, 1.4786212839507293, 1.2606657533955001, 1.1531733707480682, 1.1147308053455016, 1.058184567444851, 0.9942046301801816, 0.940069253515753]
zero = [2.302421418707068, 2.3022746671198884, 2.3021431184461356, 2.3020252225102973, 2.3019195827656724, 2.3018249417288428, 2.3017401677118525, 2.301664242743934, 2.3015962515838715, 2.301535371732133]

line1 = ax.plot(epochs, glorot, 'r-', label="Glorot")
line2 = ax.plot(epochs, normal, 'b-', label="Normal")
line3 = ax.plot(epochs, zero, 'g-', label="Zero")
lines = line1+line2+line3
labels = [l.get_label() for l in lines]
ax.legend(lines, labels)

plt.xlabel("Epoch")
plt.ylabel("Loss on training set")
plt.title("Loss on the MNIST training set over 10 epochs for 3 different \n initializations of a two hidden layer MLP")
plt.savefig("Initialization.png")


# Figure for CNN loss curve
fig = plt.figure()
ax = fig.add_subplot(111)

epochs = np.arange(1, 11)

train_loss = [0.2816159084749222, 0.13255757739067078, 0.09313151215553284, 0.07489036657810211, 0.06301878925800324, 0.05477934633255005, 0.04808822777748108, 0.042923189496994016, 0.03920403305053711, 0.03611740089893341]
valid_loss = [0.24940405373573304, 0.12286567389965057, 0.09096207387447357, 0.07692856514453889, 0.06842999408245087, 0.06271780321598053, 0.05835113306045532, 0.055349463081359865, 0.0530753485918045, 0.05160532395839691]

line1 = ax.plot(epochs, train_loss, 'b-', label='Training')
line2 = ax.plot(epochs, valid_loss, 'r-', label='Validation')
lines=line1+line2

labels = [l.get_label() for l in lines]
ax.legend(lines, labels)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves for the CNN")
plt.savefig("CNN.png")


# #Figure for MLP loss curve
fig = plt.figure()
ax = fig.add_subplot(111)

epochs = np.arange(1, 11)

train_loss = [0.33292998856309985, 0.2466289686802872, 0.19582370141002942, 0.1627267703234354, 0.13873290253349635, 0.12061145068260368, 0.1063212057824769, 0.09471232181249381, 0.08502928410266607, 0.07680919071100167]
valid_loss = [0.3024005475331972, 0.23057749735203656, 0.18846434150006472, 0.1617695133945614, 0.1432441361300729, 0.12988394939817735, 0.11971998410926019, 0.11180170433818175, 0.10552537892192843, 0.10040691141819692]

line1 = ax.plot(epochs, train_loss, 'b-', label='Training')
line2 = ax.plot(epochs, valid_loss, 'r-', label='Validation')
lines=line1+line2

labels = [l.get_label() for l in lines]
ax.legend(lines, labels)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves for the MLP")
plt.savefig("MLP.png")


# Figure for finite difference Q
fig = plt.figure()
ax = fig.add_subplot(111)

n_vals = [1,10,50,100,500,1000,10000]

max_diff = [0.0017298100871611763, 6.664431725681612e-05, -1.0110834549811876e-08, -2.527706153762632e-09, -1.012516303439126e-10, -2.543467346460826e-11, -1.6453097564927965e-12]

line = ax.plot(n_vals, max_diff, 'b-')

plt.xscale("log")
#plt.yscale("log")
plt.xlabel("Value of N")
plt.ylabel("Max difference between estimated \n and real gradient")
plt.title("Max Difference between estimated and real gradients \n against precision of weight perturbation")
plt.savefig("FiniteDiff.png")

#Figure for dropout investigation
fig = plt.figure()
ax = fig.add_subplot(111)

p_vals = [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6]

max_score = [0.9849, 0.9852, 0.9844, 0.9858, 0.9819, 0.9835, 0.9826, 0.9829]

line = ax.plot(p_vals, max_score, 'b-')

plt.xlabel("Dropout Probability")
plt.ylabel("Accuracy after 10 epochs of training")
plt.title("Effect of dropout on accuracy after 10 epochs")
plt.savefig("Reg_acc.png")

fig = plt.figure()
ax = fig.add_subplot(111)

p_vals = [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6]
max_score = [0.0501256276011467, 0.05015577538013458, 0.05235868442058563, 0.05032153260707855, 0.05927604737281799, 0.054625498700141904, 0.05834926722049713, 0.06090590441226959]

line = ax.plot(p_vals, max_score, 'b-')

plt.xlabel("Dropout probability")
plt.ylabel("Loss after 10 epochs of training")
plt.title("Effect of Dropout on loss after 10 epochs")
plt.savefig("Reg_loss.png")
