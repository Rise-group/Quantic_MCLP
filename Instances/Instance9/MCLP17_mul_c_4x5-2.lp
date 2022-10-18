\ Model MCLPmin
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  y_0 + y_1 + y_2 + y_3 + 3 y_4 + y_5 + y_6 + 3 y_7 + y_8 + y_9 + y_10
   + y_11 + y_12 + y_13 + y_14 + y_15 + y_16 + y_17 + 5 y_18 + 3 y_19
Subject To
 c1a_0: y_0 + x_0 >= 1
 c1a_1: y_1 + x_0 >= 1
 c1a_2: y_2 + x_3 >= 1
 c1a_3: y_3 + x_3 >= 1
 c1a_4: y_4 + x_3 >= 1
 c1a_5: y_5 + x_0 >= 1
 c1b_6: y_6 >= 1
 c1a_7: y_7 + x_12 >= 1
 c1a_8: y_8 + x_3 >= 1
 c1b_9: y_9 >= 1
 c1b_10: y_10 >= 1
 c1a_11: y_11 + x_12 + x_16 >= 1
 c1a_12: y_12 + x_12 >= 1
 c1a_13: y_13 + x_12 >= 1
 c1a_14: y_14 + x_19 >= 1
 c1a_15: y_15 + x_16 >= 1
 c1a_16: y_16 + x_16 >= 1
 c1a_17: y_17 + x_12 + x_16 >= 1
 c1a_18: y_18 + x_19 >= 1
 c1a_19: y_19 + x_19 >= 1
 c2_19: x_0 + x_3 + x_12 + x_16 + x_19 = 2
Bounds
Binaries
 y_0 y_1 y_2 y_3 y_4 y_5 y_6 y_7 y_8 y_9 y_10 y_11 y_12 y_13 y_14 y_15 y_16
 y_17 y_18 y_19 x_0 x_3 x_12 x_16 x_19
End
