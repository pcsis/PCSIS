<div align="center">
  	<h1>
    	PCSIS
  	</h1>
</div>

This is the repository of the code for the paper **Convex Computations for Controlled Safety Invariant Sets of Black-box Discrete-time Dynamical Systems**.



## 1. Introduction

*PCSIS* is a tool designed for synthesizing control safe invariant sets (CSIS) for black-box discrete-time systems. Since the system's underlying model is unknown and only limited simulation data are available, PCSIS synthesizes a PAC CSIS, where the probability that a control input exists to keep the system within the CSIS at the next time step is evaluated, with a predefined level of confidence. If the system successfully remains within the set at the next time step, one can then reapply the invariance evaluation to the new state, thereby facilitating a recursive assurance of invariance and safety. The tool constructs the PAC CSIS from a dataset using a linear programming approach and employs **[Gurobi](https://www.gurobi.com/downloads/)**, a state-of-the-art solver, to solve the linear programming. Beyond PAC CSIS synthesis, PCSIS offers features such as visualization, estimation of SIS using the Monte Carlo method, and other utilities. Once a PAC CSIS is successfully synthesized, PCSIS also supports controller fitting through polynomial regression or neural networks.



## 2. Installation

1. Install [Gurobi](https://www.gurobi.com/downloads/), noting that a license is required. An academic license can be obtained by applying [here](https://www.gurobi.com/features/academic-named-user-license/). The version of Gurobi Optimizer used in this paper is 11.0.3.



2. Run the following commands to create an environment and install the necessary dependencies. Python version 3.8 is recommended. 

```bash
conda create --name pcsis python=3.8

conda activate pcsis

python -m pip install .
```



## 3. How to run

The `benchmarks` folder contains two subfolders: `sis` and `csis`. The `sis` folder provides benchmarks for calculating PAC SIS on black-box systems without control inputs, while the `csis` folder includes benchmarks for calculating PAC CSIS on black-box systems with control inputs. The following commands illustrate how to quickly run the provided benchmarks:

- To synthesize PAC SIS, an example is:

```bash
python .\Benchmarks\sis\vanderpol_sis.py
```

- To synthesize PAC CSIS, an example is:

```bash
python .\Benchmarks\csis\pendulum_csis.py
```



New system models can be added in the `pcsis/model` folder. Additionally, PCSIS supports synthesizing PAC CSIS directly from sampled data.



In each benchmark, parameters can be manually adjusted as required. The parameter `degree_poly` specifies the degree of the polynomial $h_1(x)$, while `random_seed` sets the random number seed for reproducibility. The parameters `lamda`, `alpha`, `beta`, `N`, `M`, `N1`, `U_al`, `epsilon`, `epsilon_prime`, `K`, and `K_prime` correspond to $\lambda$, $\alpha$, $\beta$, $N$, $M$, $N^\prime$, $U_{al}$, $\epsilon$, $\epsilon^\prime$, $K$, and $K^\prime$ as defined in the paper.



