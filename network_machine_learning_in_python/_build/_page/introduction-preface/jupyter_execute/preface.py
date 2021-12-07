#!/usr/bin/env python
# coding: utf-8

# # Front Matter

# ## Abstract

# This thesis is essentially a general overview of spectral methods on networks, and how you can use tools from a network's eigenspace to understand and explain the network more deeply. Why are networks an interesting thing to learn about, and why should you care?
# 
# Well, at some level, every aspect of reality seems to be made of interconnected parts. Atoms and molecules are connected to each other with chemical bonds. Your neurons connect to each other through synapses, and the different parts of your brain connect to each other through groups of neurons interacting with each other. At a larger level, you are interconnected with other humans through social networks, and our economy is a global, interconnected trade network. The Earth's food chain is an ecological network, and larger still, every object with mass in the universe is connected to every other object through a gravitational network.
# 
# So if you can understand networks, you can understand a little something about everything!
# 
# We'll cover the fundamentals of spectral methods with respect to network data science, focusing on developing intuition on networks as statistical objects, while paired with relevant Python code. By the end of this thesis, you will be able to utilize efficient and easy to use tools available for performing analyses on networks. You will also have a whole new range of statistical techniques in your toolbox, such as representations, theory, and algorithms for networks.
# 
# We'll spend this thesis learning about network algorithms by showing how they're implemented in production-ready Python frameworks:
# - Numpy and Scipy are used for scientific programming. They give you access to array objects, which are the main way we'll represent networks computationally.
# - Scikit-Learn is very easy to use, yet it implements many Machine Learning algorithms efficiently, so it makes a great entry point for downstream analysis of networks.
# - Graspologic is an open-source Python package developed by Microsoft and the NeuroData lab at Johns Hopkins University which gives you utilities and algorithms for doing statistical analyses on network-valued data.
# 
# The thesis favors a hands-on approach, growing an intuitive understanding of networks through concrete working examples and a bit of theory. While you can read this thesis without picking up your laptop, I highly recommend you experiment with the code examples available online as Jupyter notebooks at [http://docs.neurodata.io/graph-stats-book/index.html](http://docs.neurodata.io/graph-stats-book/index.html).
# 
# **Primary Reader and Advisor**: Joshua Vogelstein  
# **Secondary Reader**: Avanti Athreya

# ## Acknowledgements

# Big thanks to everybody who has been reading the thesis as I write and giving feedback. This list includes Dax Pryce, Ross Lawrence, Geoff Loftus, Alexandra McCoy, Olivia Taylor Peter Brown, Sambit Panda, Eric Bridgeford, Josh Vogelstein, and Ali Sad-Aldin. 
# 
# I am grateful to my advisor, Joshua Vogelstein, for his insights and strong feedback. The value he puts on clarity and simplicity in any mathematical model has been an enormous help throughout this process.
# 
# I am also especially grateful to Eric Bridgeford, who has been giving me constant feedback throughout the writing process. I would be lost in a sea of papers without his help.

# ## **Dedication**
# 
# This thesis is dedicated to my father, Geoffrey Loftus, for teaching me the value of rigor in science and for being a resoundingly positive role model throughout my life, and to my mother, Susan Loftus, for teaching me to never give up in the face of adversity.

# ## Contents

# **Abstract**  
# **Acknowledgements**  
# **List of Figures**  
# **1. Matrix Representations of Networks**  
#   The Adjacency Matrix  
#   The Incidence Matrix  
#   The Oriented Incidence Matrix  
#   The Degree Matrix  
#   The Laplacian Matrix  
# **2. Why Embed Networks?**  
#   High Dimensionality of Network Data  
#   Latent Estimation  
#   The Latent Position Matrix  
#   Edge Probability Estimation
#   Block Probability Matrices  
#   Geometry of Latent Positions  
# **3. Spectral Embedding Methods**  
#   Singular Vectors and Singular Value Decomposition  
#   Breaking Down the Laplacian  
#   Matrix Rank  
#   Sums of Rank One Matrices  
#   Laplacian Approximation Through Summation  
#   Increased Usefulness of Approximation with Larger Networks  
#   Matrix Rank and Spectral Embedding  
#   Dimensionality Estimation  
#   The Two-Truths Phenomenon  
# **4. Multiple-Network Representation Learning**  
#   Data Generation  
#   Simple Embedding Methods on Multiple Networks  
#   Averaging Separately  
#   Averaging Together  
#   Different Types of Multi-Network Representation Learning  
#   Network Combination: Together  
#   Network Combination: Separate  
#   Embedding Combination  
#   Multiple-Adjacency Spectral Embedding  
#   Overview of MASE
#   Data Generation  
#   Embedding  
#   Combining Embeddings  
#   Joint Embedding of Combinations  
#   Score Matrices  
#   Omnibus Embedding  
#   OMNI on Four Networks  
#   Overview of OMNI  
#   The Omnibus Matrix  
#   Embedding the Omnibus Matrix  
#   Using the Omnibus Embedding  
# **5. Joint Representation Learning**  
#   Data Generation  
#   Covariates  
#   Covariate-Assisted Spectral Embedding  
#   Weight Exploration  
#   Weight Estimation  
#   Omnibus Joint Embedding  
#   MASE Joint Embedding  
# **6. Single-Network Vertex Nomination**  
#   Spectral Vertex Nomination  
#   Finding a Single Set of Nominations  
#   Nominations for Each Node  
# **7. Out-of-Sample Embedding**  
#   Data Generation  
#   Probability Vector Estimation  
#   Inversion of Probability Vector Estimation  
#   The Moore-Penrose Pseudoinverse  
#   Using the Pseudoinverse for Out-of-Sample Estimation  
# **8. Anomaly Detection for Timeseries of Networks**  
#   Simulating Timeseries Data  
#   Approaches for Anomaly Detection  
#   Detecting if the First Time-Point is an Anomaly  
#   Hypothesis Testing a Test Statistic  
#   Bootstrapped Distribution Estimation  
#   P-Value Estimation  
#   Testing the Remaining Time-Points  
#   The Distribution of the Bootstrapped Test Statistic  

# ## List of Figures

# 1.1 A Three-Node Network  
# 1.2 Adjacency Matrix and Layout Plot  
# 1.2 Incidence Matrix and Layout Plot  
# 1.3 Oriented Incidence Matrix and Layout Plot  
# 2.1 Euclidean data represented as a data matrix and represented in Euclidean space  
# 2.2 Clustered data after K-Means  
# 2.3 A Network With Two Groups  
# 2.4 Latent Position Estimation  
# 2.5 Estimated and True Block Probability Matrices  
# 2.6 Geometry of Latent Positions  
# 3.1 The Spectral Embedding Algorithm  
# 3.2 A Simple Network  
# 3.3 The Laplacian Is Just a Function of the Adjacency Matrix  
# 3.4 Decomposing our Simple Laplacian into Eigenvectors and Eigenvalues with SVD  
# 3.5 We Can Recreate our Simple Laplacian by Summing All the Low-Rank Matrices 
# 3.6 The Sum of an Increasing Number of Low-Rank Matrices  
# 3.7 Summing Only Two Low-Rank Matrices Approximates the Normalized Laplacian  
# 3.8 The Latent Position Matrix  
# 3.9 Low-Rank Matrices Contain the Same Information As Columns of the Latent Position Matrix  
# 3.10 Expressing the Sum With Columns of the Latent Position Matrix  
# 3.11 Scree Plot  
# 3.12 The Adjacency Spectral Embedding  
# 3.13 The Laplacian Spectral Embedding  
# 3.14 Affinity vs Core-Periphery Structure  
# 4.1 Different Sets of Brain Networks  
# 4.2 Averaged Embedded Networks  
# 4.3 Clustering with GMM  
# 4.4 Embedding when we Average Everything Together  
# 4.5 Network Comparison
# 4.6 Averaged Brain Network  
# 4.7 Combined Network Group Embedding  
# 4.8 Separate Network Group Embedding  
# 4.9 Combined Embedding  
# 4.10 MASE Embedding On Network Groups  
# 4.11 Four Different Networks  
# 4.12 ASE on Four Networks  
# 4.13 Latent Position Matrices for Four Embeddings  
# 4.14 Combined Embeddings For Four Networks  
# 4.15 Visualizing the Joint Embedding  
# 4.16 MASE Embedding  
# 4.17 Score Matrices and Edge Probabilities  
# 4.18 ASE for Four Networks  
# 4.19 Omnibus Embedding for Four Networks  
# 4.20 The Omnibus Matrix for Two Networks  
# 4.21 Full Omnibus Matrix for All Four Networks  
# 4.22 Latent Positions for the Omnibus Embedding in Matrix Form  
# 4.23 Latent Positions for the Omnibus Embedding in Euclidean Space  
# 4.24 Mouse Networks Corresponding to a Single Node after Omnibus Embedding
# 5.1 Stochastic Block Model with Three Communities  
# 5.2 Laplacian-Embedded Latent Positions  
# 5.3 Covariate Visualization  
# 5.4 Laplacian and Covariates  
# 5.5 Embedding Without Weights  
# 5.6 Comparison of Embeddings for Different Weights on $YY^\top$  
# 5.7 Embedding With Weights  
# 5.8 The Benefit Of Using Two Types of Information  
# 5.9 Embedding with Graspologic  
# 5.10 Omni Embedding for Topology and Covariates  
# 5.11 MASE Joint Embedding  
# 6.1 Latent Positions and Seeds for an SBM With Three Communities  
# 6.2 Centroid for Seed Latent Positions  
# 6.3 Nomination List  
# 6.4 Nomination List: Network Plot  
# 6.5 Nominations for Each Seed Node  
# 7.1 Adjacency Matrix and Vector For Additional Node  
# 7.2 Latent Positions for Original Network  
# 7.3 Estimated Probability Vector for First Node  
# 7.4 A Noninvertible Linear Transformation  
# 7.5 The Best Approximation The Pseudoinverse Can Do  
# 7.6 Estimating the Out-of-Sample Latent Position  
# 7.7 Latent Positions with Out-of-Sample Estimate  
# 8.1 Network Timeseries Data  
# 8.2 Distribution of test statistics with the same latent positions  
# 8.3 Test Statistics for Each Timeseries  
# 8.4 Distribution of test statistics with the same latent positions  
# 8.5 Distribution Comparison for Bootstrapped and True Test Statistic
