principle = 0
rate = 0
time = 0

while principle <= 0:
    principle = float(input("Enter the initial principal amount: "))
    if principle <= 0:
        print("Invalid principal amount. Please enter a positive value.")

while rate <= 0:
    rate = float(input("Enter the interest rate : "))
    if rate <= 0:
        print("Invalid interest rate. Please enter a positive value.")
 
while time <= 0:
    time = int(input("Enter the time in years: "))
    if principle <= 0:
        print("Invalid time. Please enter a positive value.")

print(principle)
print(rate)
print(time)
A = principle * pow((1 + rate/100), time)

print(f"Balance after {time} years is ${A: .2f}")

# Formula for calculating compound interest: A = P(1 + r/n)^(nt)
