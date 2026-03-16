import random
import math
from statistics import mean

# This script is used to find parameters for SHELL that can support a given
# number of plaintext bits, noise bits, and multiplication depth. The script
# will build a modulus chain where the first primes are used to hold the noise
# and the last primes are used to support the `rescale` operation, all roughly
# nearby the desired scaling factor. The script will also estimate the maximum
# error introduced to the plaintext by differences between the true scaling
# factor and the found factors during rescaling.

log_n = 11
plaintext_bits = 48  # must be large enough to hold the plaintext * scaling_factor**2

mul_depth = 1
# For reference, IEEE 754:
#   16-bit (half precision) has 10 bits of mantissa.
#   32-bit (single precision) has 23 bits of mantissa.
#   64-bit (double precision) has 52 bits of mantissa.
# A good way to explore possible scaling factors is to set it very large.
# This script will look from 0 - 2*desired_scaling_factor for primes to support
# multiplication, starting at the desired_scaling_factor and moving outwards.
desired_scaling_factor = 2**13

total_noise_bits = 70


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


print(f"Total plaintext bits: {plaintext_bits}\n")
assert plaintext_bits <= 58, "Total number of plaintext bits must be <= 58"
assert desired_scaling_factor**2 <= 2**plaintext_bits, "Scaling factor too large."
assert desired_scaling_factor >= 1, "Scaling factor too small."

# Find a prime that is large enough to hold the plaintext.
plaintext_modulus = find_prime_mod_2n(plaintext_bits)
assert plaintext_modulus is not None, "Could not find plaintext modulus."
print(f"Plaintext modulus: {plaintext_modulus}\n")

# Find primes large enough to hold both the noise and the plaintext.
total_bits = plaintext_bits + total_noise_bits
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
first_prime_num_bits = max(total_bits, plaintext_bits + 1)
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


if mul_depth > 0 and desired_scaling_factor > 1:
    # Now find primes to support ModReduce used in multiplication. The primes should
    # be around the scaling factor and do not need need to be larger than the
    # plaintext modulus.
    mul_primes = []

    # Ideally, the prime p == scaling factor but this is not necessarily
    # possible. The should be as close as possible to the scaling factor so
    # search for primes outward from the scaling factor.
    for i in range(1, desired_scaling_factor - 3):
        p_pos = desired_scaling_factor + i
        p_neg = desired_scaling_factor - i

        if check_prime(p_pos, 10):
            if p_pos % two_n == 1 and p_pos not in found_primes:
                mul_primes.append(p_pos)
                if len(mul_primes) == mul_depth:
                    break
        if check_prime(p_neg, 10):
            if p_neg % two_n == 1 and p_neg not in found_primes:
                mul_primes.append(p_neg)
                if len(mul_primes) == mul_depth:
                    break

    if len(mul_primes) != mul_depth:
        raise ValueError("Could not find enough multiplication primes.")

    print(f"Found multiplication primes {mul_primes}")
    print(f"A good scaling factor for these primes is {mean(mul_primes)}")
    scaling_factor = mean(mul_primes)

    # The maximum error introduced by the prime p != scaling factor is
    max_plaintext = plaintext_modulus // 2
    max_scaled_plaintext = max_plaintext // (scaling_factor * scaling_factor)
    print(f"Max plaintext which can be represented: {max_scaled_plaintext}")

    operand = max_scaled_plaintext ** (1.0 / (mul_depth + 1))
    operand *= scaling_factor * scaling_factor

    for p in mul_primes[::-1]:
        # Multiplication first does ModReduce which divides by p.
        operand = round(operand / p)
        # Then multiplication.
        operand = operand * operand
        # Now if p == scaling_factor, operand is back to being scaled by the
        # scaling factor**2. But if not, some error will be introduced.

    # After all the multiplications, the scaling factor is divided out.
    operand = operand / (scaling_factor * scaling_factor)

    max_error = abs(max_scaled_plaintext - operand)
    print(
        f"Estimated max error introduced by the mod-switching RNS primes: {max_error}"
    )
    print(
        f"Estimated max error as a fraction of the maximum representable value: {max_error / max_scaled_plaintext}"
    )

    # Add the mul_primes in reverse to the found_primes list so that earlier
    # operations introduce less noise. (The ModReduce operation pulls from the
    # back).
    found_primes.extend(mul_primes[::-1])

else:
    scaling_factor = desired_scaling_factor
    max_scaled_plaintext = plaintext_modulus // (scaling_factor * scaling_factor)
    max_error = 0


print(
    "\nExample configuration:\n\n"
    f"# Num plaintext bits: {plaintext_bits}, noise bits: {total_noise_bits}\n"
    f"# Max plaintext value: {max_scaled_plaintext}, est error: {(max_error/max_scaled_plaintext):.3f}%\n"
    f"context = hadal_flow.create_context64(\n"
    f"    log_n={log_n},\n"
    f"    main_moduli={found_primes},\n"
    f"    plaintext_modulus={plaintext_modulus},\n"
    f"    noise_variance=8,\n"
    f"    scaling_factor={scaling_factor},\n"
    f"    seed='',\n"
    f")"
)
