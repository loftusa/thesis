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
# 
# Lastly, I'm grateful to my father, Geoffrey Loftus, for teaching me the value of rigor in science and for being a resoundingly positive role model throughout my life.

# ## **Dedication**
# 
# This thesis is dedicated to my father, Geoffrey Loftus, for teaching me the value of rigor in science and for being a resoundingly positive role model throughout my life, and to my mother, Susan Loftus, for teaching me to never give up in the face of adversity.

# ## Contents

# **Abstract**  
# **Acknowledgements**  
# **List of Figures**  
# **Matrix Representations of Networks**
#   The Adjacency Matrix  
#   The Incidence Matrix
# **Why Embed Networks?**  
# **Spectral Embedding Methods**  
# **Multiple-Network Representation Learning**  
# **Joint Representation Learning**  
# **Single-Network Representation Learning**  
# **Out-of-Sample Embedding**  
# **Anomaly Detection for Timeseries of Networks**  
# 

# This thesis is organized into three parts. 
# 
# Part I, Foundations, gives you a brief overview of the kinds of things you'll be doing in this thesis, and shows you how to solve a network data science problem from start to finish. It covers the following topics:
# - What a network is and where you can find networks in the wild
# - All the reasons why you should care about studying networks
# - Examples of ways you could apply network data science to your own projects
# - An overview of the types of problems Network Machine Learning is good at dealing with
# - The main challenges you'd encounter if you explored Network Learning more deeply
# - Exploring a real network data science dataset, to get a broad understanding of what you might be able to learn.
# 
# Part II, Representations, is all about how we can represent networks statistically, and what we can do with those representations. It covers the following topics:
# - Ways you can represent individual networks
# - Ways you can represent groups of networks
# - The various useful properties different types of networks have
# - Types of network representations and why they're useful
# - How to represent networks as a bunch of points in space
# - How to represent multiple networks
# - How to represent networks when you have extra information about your nodes
# 
# Part III, Applications, is about using the representations from Part II to explore and exploit your networks. It covers the following topics:
# - Figuring out if communities in your networks are different from each other
# - Selecting a reasonable model to represent your data
# - Finding nodes, edges, or communities in your networks that are interesting
# - Finding time points which are anomalies in a network which is evolving over time
# - What to do when you have new data after you've already trained a network model
# - How hypothesis testing works on networks
# - Figuring out which nodes are the most similar in a pair of networks

# ## List of Figures

# In[ ]:




