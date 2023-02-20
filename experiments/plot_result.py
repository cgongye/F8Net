# %%
import numpy as np
import matplotlib.pyplot as plt
with open("result.txt", "r") as f:
    lines = f.readlines()
def parse_line(line):
    line = line.split(',')
    return int(line[0]), float(line[1])
result = list(map(parse_line,lines))
result = np.array(result)
# %%
print(result)
# %%
plt.scatter(range(64),result[:,1])
# %%
weight = np.load("weight.npy")
bias = np.load("bias.npy")
# %%
print(weight.shape)
# %%
std = np.std(weight,axis=(1,2,3))
abs_avg = np.average(np.abs(weight),axis=(1,2,3))
avg = np.average(weight,axis=(1,2,3))
pop_count = (weight!=0)
print(pop_count.shape)
pop_count = np.sum(pop_count,axis=(1,2,3))
plt.figure()
plt.title("Per-channel bit-flip accuracy vs standard diviation of weights")
plt.scatter(std, result[:,1])
plt.xlabel("Standard diviation of the weights of each output channel")
plt.ylabel("Accuracy")
plt.show()
plt.close()
# %%
plt.figure()
# plt.title("Per-channel bit-flip accuracy vs standard diviation of weights")
plt.scatter(abs_avg,std)
plt.ylabel("Standard diviation of the weights of each output channel")
plt.xlabel("Average of the absoluate value of the weights of each output channel")
plt.show()
plt.close()
# %%
