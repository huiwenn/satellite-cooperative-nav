# Satellite Cooperative Navigation

A simple multi-agent particule world with continuous observation and discrete action spaces that relies on the Chlohessy Wiltshire equations for modeling the physics between agents.

# Coding Structure
multiagent/environment.py: contains code for environment simulation

multiagent/core.py: contains classes for various objects (Entities, landmarks, agents) that are used throughout the code

multiagent/scenario.py: contains base scenario object that is used to extend the navigation environment

# Cooperative Navigation Environment Description
Each satellite is tasked with rendezvousing to an assigned goal location by the end of the episode. This task description is meant to emulate close proximity satellite operations, where individual satellites need to navigate to a goal (also called a rendezvous) without collision.
