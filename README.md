# Satellite Cooperative Navigation

A simple multi-agent particule world with continuous observation and discrete action spaces that relies on the Chlohessy Wiltshire equations for modeling the physics between agents. This environment was adapted froom OpenAI’s Multi-Particle Environment (MPE) codebase. 



# Coding Structure
multiagent/environment.py: contains code for environment simulation

multiagent/core.py: contains classes for various objects (Satellites, landmarks, agents) that are used throughout the code

multiagent/scenario.py: contains base scenario object that is used to extend the navigation environment

# Cooperative Navigation Environment Description
Each satellite is tasked with rendezvousing to an assigned goal location by the end of the episode. This task description is meant to emulate close proximity satellite operations, where individual satellites need to navigate to a goal (also called a rendezvous) without collision. The image below contains the example rendezvous scenario contained in this environment. 

![alt text](https://github.com/sydneyid/satellite-cooperative-nav/blob/main/images/navigation_task.png)


# Citation

If you found this codebase useful in your research, please consider citing

```bibtex
@article{Dolan2023,
title={Satellite Navigation and Coordination with Limited Information Sharing},
journal={Learning for Dynamics and Control Conference},
publisher ={Proceedings of Machine Learning Research},
 author={Dolan, Sydney and Nayak, Siddharth and Balakrishnan, Hamsa},
 year={2023}, pages={1058–1071}} 
```
