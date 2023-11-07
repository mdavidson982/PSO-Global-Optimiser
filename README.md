Structure of various variables:

upper_bound:
[num_dim x 1] matrix.  Stores the values that maintain the upper bound of the domain

lower_bound:
[num_dim x 1] matrix.  Stores the values htat maintain the lower bound of the domain

pos_matrix:
[num_dim X num_part] matrix.  Every column is a particle, and every row is a dimension.  So, every column has the x, y, z etc. coordinates of the particle.


vel_matrix:
[num_dim X num_part] matrix.  

p_best:
[num_dim + 1 X num_part] matrix.  Stores the coordinates of every particle's personal best, along with the evaluated result on the bottom. So,

    x1: | | | |   <- this upper part has the same dimensions as the pos and vel matrices
    x2: | | | |
    ... 
    xn: | | | |
        _______
result: | | | |

g_best:
[num_dim X 1] matrix.  Stores the global best

v_max:
[num_dim X 1] matrix.  Maximum velocity of any given particle, given by alpha(U - L).