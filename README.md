# Jacks Car Rental
This is an implementation of Example 4.2 from the Book 'Reinforcement Learning: An Introduction' by Richard Sutton and Andrew Barto.
Unlike some other implementations that I have seen, I specified the environment dynamics separately as 3-argument p and expected reward function.
The core of the three-argument p is implemented in C as it would otherwise take very long to run.

# Setup
```bash
pip install matplotlib scipy numpy
cc -fPIC -shared -O3 -o libp3.so p3arg.c
python jacks_car_rental.py
```

# Environment Specification
The three-argument p was derived as such:

For any of the two locations, the following variables are introduced:
- `c` is the number of cars there at the start of the day (*after* the action was taken)
- `c'` is the number of cars there are at the end of the day
- `l1` is the number of rental requests (with corresponding Poisson parameter)
- `l2` is the number of returned cars on that day

We are looking for the probability mass function `p(c' | c)`, as it denotes the probability of reaching the partial state `c'`, after starting the day with `c` cars.
To get there, another variable `b` is introduced which denotes how many cars are at the location after all requests have entered.

Now, using the number of requests `l1`, we can model the pmf of `b` given `c`:

`p(b | c) = `
- `Pr(l1 >= c)`, if `b = 0`
- `Pr(l1 = (c-b))`, else

And the pmf of `c'` given `b`, using the number of returns `l2`:

`p(c' | b) = `
- `Pr(l2 >= (c' - b))`, if `c' = 20`
- `Pr(l2 = (c'-b))`, else

Where in both cases `Pr` is the pmf or cdf of the Poisson variable `l1` or `l2`.

The pmf`s can be understood as follows:
- In the first case, to be left with `b = 0` cars after all requests have been made means, that *at least* `c` cars have been requested. Otherwise, if some cars are left over, exactly `c - b` cars have had to been requested.
- In the second case, in order to be left with `c' = 20` cars at the end of the day, *at least* `c' - b` cars have to be returned. Otherwise, if no cars were returned superflously, exactly `c' - b` cars were returned.

Proceeding onwwards, we can now model the desired distribution:

`p(c', b | c) = p(c' | b) * p(b | c)`

and finally, by marginalizing the above:

`p(c' | c) = Σ_b p(c', b | c)`

The three argument follows by the independence of the business at the two locations:

`p( (c1', c2') | (s1, s2), a) = p(c1' | s1 - a)*p(c2' | s2 + a)`
