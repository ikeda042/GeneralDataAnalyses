import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set()

def box_plot_function(data:list[np.ndarray] | list[float|int], 
                      labels:list[str], 
                      xlabel:str, 
                      ylabel:str, 
                      save_name:str) -> None:
    fig = plt.figure(figsize=[10,7])
    plt.boxplot(data,sym="")
    for i, d in enumerate(data, start=1):
        x = np.random.normal(i, 0.04, size=len(d)) 
        plt.plot(x, d, 'o', alpha=0.5)  
    plt.xticks([i+1 for i in range(len(data))], [f"{i}" for i in labels])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    fig.savefig(f"{save_name}.png",dpi = 500)


data1 = np.random.normal(0, 1, size=100)
data2 = np.random.normal(0, 2, size=100)
data3 = np.random.normal(0, 3, size=100)

data  = [data1, data2, data3]
labels = ["data1", "data2", "data3"]
xlabel = "Samples"
ylabel = "Lengths (pixel)"
save_name = "result"
box_plot_function(data, labels, xlabel, ylabel, save_name)
