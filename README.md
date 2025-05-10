# Jacks Car Rental
This is an implementation of Example 4.2 from the Book 'Reinforcement Learning: An Introduction' by Richard Sutton and Andrew Barto.
Unlike some other implementations that I have seen, I specified the environment dynamics separately as 3-argument p and expected reward function.
The core of the three-argument p is implemented in C as it would otherwise take very long to run.

# Setup
```bash
pip install matplotlib
cc -fPIC -shared -O3 -o libp3.so p3arg.c
python jacks_car_rental.py
```

# Environment Specification
The three-argument p was derived as such:
