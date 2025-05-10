#include <math.h>
#define E 2.718281828

#include <stdio.h>
double pdf(double k, double lam) {
	if (k < 0)
		return 0;
	double lam_k = pow(lam, k);
	double fac_k = tgamma(k+1);
	double e_nlam = exp(-lam);
	return (lam_k*e_nlam)/fac_k;
}

double cdf(double k, double lam) {
	double r = 0;
	for (int i = 0; i <= k; i++) {
		r += pdf(i, lam);
	}
	return r;
}

double b_given_c(int b, int c, int mean_requests) {
	if (b == 0) {
		if (c<=0) {
			return 1;
		}
		return 1 - cdf(c - 1, (double)mean_requests);
	}
	return pdf(c - b, (double)mean_requests);
}

double c_prime_given_b(int c_prime, int b, int mean_returns, int MAX) {
	if (c_prime == MAX) {
		if ((c_prime - b) <= 0) {
			return 1;
		}
		return 1 - cdf(c_prime - b - 1, (double)mean_returns);
	}
	return pdf(c_prime - b, (double)mean_returns);
}

double c_prime_given_c(int c_prime, int c, int mean_requests, int mean_returns, int MAX) {
	double s = 0;
	for (int i = 0; i <= c; i++) {
		s += c_prime_given_b(c_prime, i, mean_returns, MAX) * b_given_c(i, c, mean_requests);
	}
	return s;
}
