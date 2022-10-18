\ Model MCLPmin
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  3 y_0 + 3 y_1 + y_2 + 2 y_3 + 3 y_4 + 3 y_5 + 2 y_6 + 3 y_7 + 4 y_8
   + 3 y_9 + y_10 + 2 y_11 + 4 y_12 + 5 y_13 + 5 y_14 + 2 y_15
Subject To
 c1a_0: y_0 + x_0 >= 1
 c1a_1: y_1 + x_0 >= 1
 c1b_2: y_2 >= 1
 c1a_3: y_3 + x_7 >= 1
 c1a_4: y_4 + x_0 >= 1
 c1b_5: y_5 >= 1
 c1a_6: y_6 + x_7 >= 1
 c1a_7: y_7 + x_7 >= 1
 c1a_8: y_8 + x_12 >= 1
 c1b_9: y_9 >= 1
 c1b_10: y_10 >= 1
 c1a_11: y_11 + x_7 >= 1
 c1a_12: y_12 + x_12 >= 1
 c1a_13: y_13 + x_12 >= 1
 c1b_14: y_14 >= 1
 c1b_15: y_15 >= 1
 c2_15: x_0 + x_7 + x_12 = 2
Bounds
Binaries
 y_0 y_1 y_2 y_3 y_4 y_5 y_6 y_7 y_8 y_9 y_10 y_11 y_12 y_13 y_14 y_15 x_0
 x_7 x_12
End
