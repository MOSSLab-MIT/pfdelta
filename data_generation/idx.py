# Branch matrix indices
F_BUS = 0       # "From" bus index
T_BUS = 1       # "To" bus index
BR_R = 2        # Branch resistance
BR_X = 3        # Branch reactance
BR_B = 4        # Total line charging susceptance
RATE_A = 5      # Rate A (MVA limit)
RATE_B = 6      # Rate B (MVA limit)
RATE_C = 7      # Rate C (MVA limit)
TAP = 8         # Transformer off-nominal turns ratio
SHIFT = 9       # Transformer phase shift angle (degrees)
BR_STATUS = 10  # Branch status (1 = in-service, 0 = out-of-service)
ANGMIN = 11     # Minimum angle difference (rad)
ANGMAX = 12     # Maximum angle difference (rad)

# Power flow variables
PF = 13         # Real power flow at "from" bus (MW)
QF = 14         # Reactive power flow at "from" bus (MVar)
PT = 15         # Real power flow at "to" bus (MW)
QT = 16         # Reactive power flow at "to" bus (MVar)

# Lagrange multipliers (shadow prices)
MU_SF = 17      # Kuhn-Tucker multiplier on MVA limit at "from" bus
MU_ST = 18      # Kuhn-Tucker multiplier on MVA limit at "to" bus
MU_ANGMIN = 19  # Kuhn-Tucker multiplier lower angle limit
MU_ANGMAX = 20  # Kuhn-Tucker multiplier upper angle limit

# Bus matrix indices
BUS = 0
BUS_TYPE = 1 
PD = 2          # Active power demand
QD = 3          # Reactive power demand  
VMAX = 11
VMIN = 12


# Generator matrix indices
GEN_BUS = 0     # Bus index where the generator is located
PG = 1          # Active power generation
QG = 2          # Reactive power generation
QMAX = 3
QMIN = 4
PMAX = 8        # Maximum generator output
PMIN = 9        # Minimum generator output