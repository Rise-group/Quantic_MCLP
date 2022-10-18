\ Model MCLPmin
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  4 y_0 + 5 y_1 + 2 y_2 + 2 y_3 + y_4 + 4 y_5 + 2 y_6 + 2 y_7 + y_8
Subject To
 c1a_0: y_0 + x_0 >= 1
 c1a_1: y_1 + x_0 + x_2 >= 1
 c1a_2: y_2 + x_2 >= 1
 c1a_3: y_3 + x_0 + x_6 >= 1
 c1b_4: y_4 >= 1
 c1a_5: y_5 + x_2 >= 1
 c1a_6: y_6 + x_6 >= 1
 c1a_7: y_7 + x_6 >= 1
 c1b_8: y_8 >= 1
 c2_8: x_0 + x_2 + x_6 = 2
Bounds
Binaries
 y_0 y_1 y_2 y_3 y_4 y_5 y_6 y_7 y_8 x_0 x_2 x_6
End
