# Multithreaded-Reinforcement-Learning-A3C
Multithreaded Reinforcement Learning using A3C

## Multithreaded Architecture
This is the structure of our AC3 algorigthm

![Lazy Programmer](/readme_images/multithread)

Threaded workers copy weights from global network and run multiple episodes in paralel.

## Neural Netwrok Architecture
Shared Covolutional Neural Netwroks allow the AI to see 4 frames into the past and make decisions based solely on this. The CNNs are shared and seperate fully connected layers are used for Policy and Value networks.
![Lazy Programmer](/readme_images/nets)

## Results on Atari:
After just a few thousand episodes, the algorithm has learned to play the game proficiently: 
![Lazy Programmer](/readme_images/results)