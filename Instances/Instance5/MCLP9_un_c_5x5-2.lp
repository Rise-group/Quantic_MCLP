\ Model MCLPmin
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  5 y_0 + y_1 + y_2 + y_3 + y_4 + 2 y_5 + y_6 + y_7 + y_8 + y_9 + 5 y_10
   + y_11 + y_12 + y_13 + y_14 + y_15 + y_16 + y_17 + y_18 + y_19 + y_20
   + 4 y_21 + y_22 + 3 y_23 + y_24
Subject To
 c1b_0: y_0 >= 1
 c1a_1: y_1 + x_6 >= 1
 c1a_2: y_2 + x_7 >= 1
 c1b_3: y_3 >= 1
 c1b_4: y_4 >= 1
 c1a_5: y_5 + x_6 >= 1
 c1a_6: y_6 + x_6 + x_7 >= 1
 c1a_7: y_7 + x_6 + x_7 >= 1
 c1a_8: y_8 + x_7 >= 1
 c1b_9: y_9 >= 1
 c1b_10: y_10 >= 1
 c1a_11: y_11 + x_6 >= 1
 c1a_12: y_12 + x_7 >= 1
 c1b_13: y_13 >= 1
 c1b_14: y_14 >= 1
 c1b_15: y_15 >= 1
 c1a_16: y_16 + x_21 >= 1
 c1b_17: y_17 >= 1
 c1b_18: y_18 >= 1
 c1b_19: y_19 >= 1
 c1a_20: y_20 + x_21 >= 1
 c1a_21: y_21 + x_21 >= 1
 c1a_22: y_22 + x_21 >= 1
 c1b_23: y_23 >= 1
 c1b_24: y_24 >= 1
 c2_24: x_6 + x_7 + x_21 = 2
Bounds
Binaries
 y_0 y_1 y_2 y_3 y_4 y_5 y_6 y_7 y_8 y_9 y_10 y_11 y_12 y_13 y_14 y_15 y_16
 y_17 y_18 y_19 y_20 y_21 y_22 y_23 y_24 x_6 x_7 x_21
End
