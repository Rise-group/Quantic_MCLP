\ Model MCLPmin
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2 y_0 + 2 y_1 + y_2 + 4 y_3 + 3 y_4 + 3 y_5 + 4 y_6 + 5 y_7 + 4 y_8 + y_9
   + 2 y_10 + y_11 + 4 y_12 + 5 y_13 + 5 y_14 + 2 y_15
Subject To
 c1a_0: y_0 + x_1 >= 1
 c1a_1: y_1 + x_1 >= 1
 c1a_2: y_2 + x_1 + x_6 >= 1
 c1b_3: y_3 >= 1
 c1b_4: y_4 >= 1
 c1a_5: y_5 + x_1 + x_6 >= 1
 c1a_6: y_6 + x_6 >= 1
 c1a_7: y_7 + x_6 >= 1
 c1b_8: y_8 >= 1
 c1a_9: y_9 + x_13 >= 1
 c1a_10: y_10 + x_6 >= 1
 c1b_11: y_11 >= 1
 c1a_12: y_12 + x_13 >= 1
 c1a_13: y_13 + x_13 >= 1
 c1a_14: y_14 + x_13 >= 1
 c1b_15: y_15 >= 1
 c2_15: x_1 + x_6 + x_13 = 2
Bounds
Binaries
 y_0 y_1 y_2 y_3 y_4 y_5 y_6 y_7 y_8 y_9 y_10 y_11 y_12 y_13 y_14 y_15 x_1
 x_6 x_13
End
