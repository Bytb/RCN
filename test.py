## GIS6103 Fall 2025

#2 - Not sure how to do the one with list comprehension - doesnt make sense since you are not modifying individual elements
Q2add = ['d',1,2,'e']
Q2 = ['a', 5, 6, 'c', 3, 'b']

A3 = [x for x in Q2.extend(Q2add)]
print(A3)

# Answer 1 below
A1 = Q2 + Q2add
print(A1)

# Answer 2 below
Q2.extend(Q2add)
print(Q2)

