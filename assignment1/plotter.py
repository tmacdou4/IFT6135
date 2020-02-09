import matplotlib.pyplot as plt
import numpy as np

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
plt.show()