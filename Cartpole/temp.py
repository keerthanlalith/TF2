import numpy as np

for i in range(1, 50 + 1):
  for j in range(100):
    if (np.random.uniform(0,1)<=np.exp(-i/10)):
      print(".",end ="")
  print("") 