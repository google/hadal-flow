import random
import math
from statistics import mean

# This script is used to find parameters for SHELL that can support a given
# number of plaintext bits, noise bits, and multiplication depth. The script
# builds a modulus chain long enough to hold the noise and the plaintext, and
# computes the maximum value that can be represented in the plaintext. It
# assumes the multiplication has no rescaling operation, thus the scaling factor
# doubles with each multiplication and the plaintext modulus must be large
# enough to hold the plaintext * (scaling_factor ** (2 ** num_muls) ).

log_n = 10
plaintext_bits = 8  # not including scaling factor or mul_depth

mul_depth = 1
scaling_factor = 3

total_noise_bits = 24


# Some constants dictated by SHELL.
max_prime_bits = 58
two_n = 2 ** (log_n + 1)


# Fermat's prime test.
def check_prime(n, k):
    if n == 2:
        return True
    if not n & 1:
        return False
    for _ in range(k):
        a = random.randint(2, n - 2)
        if pow(a, n - 1, n) != 1:
            return False
    return True


def find_prime_mod_2n(bits, not_in=[]):
    for i in range(2**bits, 2 ** (bits + 2)):
        if check_prime(i, 10):
            if i % two_n == 1:
                if i not in not_in:
                    return i
    return None


total_plaintext_bits = plaintext_bits + math.ceil(
    math.log2(scaling_factor ** (2**mul_depth))
)

print(f"Total plaintext bits: {total_plaintext_bits}\n")
assert (
    total_plaintext_bits <= max_prime_bits
), f"Total number of plaintext bits must be <= {max_prime_bits}."
assert scaling_factor >= 1, "Scaling factor too small."

# Find a prime that is large enough to hold the plaintext.
plaintext_modulus = find_prime_mod_2n(total_plaintext_bits)
assert plaintext_modulus is not None, "Could not find plaintext modulus."
print(f"Plaintext modulus: {plaintext_modulus}\n")

# Find primes large enough to hold both the noise and the plaintext.
total_bits = total_plaintext_bits + total_noise_bits
num_primes_for_noise = (
    total_bits // max_prime_bits
) + 1  # Limit the primes to 58 bits.

print(f"Total bits: {total_bits}")
print(f"Num primes required to hold the noise: {num_primes_for_noise}")

found_primes = []
needed_bits_of_primes = total_bits

# The first RNS prime is special and must be larger than the plaintext modulus.
# This is because decryption ModReduces to the first RNS prime, then to the
# plaintext modulus.
first_prime_num_bits = max(total_bits, total_plaintext_bits + 1)
# The prime should be no larger than max_prime_bits.
first_prime_num_bits = min(first_prime_num_bits, max_prime_bits)
first_prime = find_prime_mod_2n(first_prime_num_bits)
assert first_prime is not None, "Could not find first noise_holding RNS prime."
found_primes.append(first_prime)
needed_bits_of_primes -= first_prime.bit_length()

# Find more primes to hold the noise, if needed. They do not need to be larger
# than the plaintext modulus.
while needed_bits_of_primes > 0:
    num_bits_needed = min(needed_bits_of_primes, max_prime_bits)
    num_bits_needed = max(num_bits_needed, 18)  # A reasonable minimum
    prime = find_prime_mod_2n(num_bits_needed, not_in=found_primes)
    assert prime is not None, "Could not find enough noise-holding RNS primes."
    found_primes.append(prime)
    needed_bits_of_primes -= prime.bit_length()

print(f"Found noise-holding primes {found_primes}\n")

max_scaled_plaintext = plaintext_modulus // (scaling_factor ** (2**mul_depth))

from operator import mul
from functools import reduce

print(
    "\nExample lattice estimator check:\n\n"
    f"from estimator import *\n"
    f"params = LWE.Parameters(n={2**log_n}, q={reduce(mul, found_primes, 1)}, Xs=ND.DiscreteGaussian(3.00), Xe=ND.DiscreteGaussian(3.00))\n"
    f"LWE.primal_bdd(params)\n"
)

print(
    "\nExample configuration:\n\n"
    f"# Num plaintext bits: {total_plaintext_bits}, noise bits: {total_noise_bits}\n"
    f"# Max representable value: {max_scaled_plaintext}\n"
    f"context = hadal_flow.create_context64(\n"
    f"    log_n={log_n},\n"
    f"    main_moduli={found_primes},\n"
    f"    plaintext_modulus={plaintext_modulus},\n"
    f"    scaling_factor={scaling_factor},\n"
    f")"
)
