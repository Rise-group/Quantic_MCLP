\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: MCLP13_Mmul_c_5x5-2

Minimize
 obj: - 65 y_0 - 65 y_1 - 65 y_2 - 65 y_3 - 65 y_4 - 65 y_5 - 65 y_6 - 65 y_7
      - 65 y_8 - 65 y_9 - 65 y_10 - 64 y_11 - 65 y_12 - 64 y_13 - 65 y_14
      - 65 y_15 - 65 y_16 - 65 y_17 - 65 y_18 - 65 y_19 - 65 y_20 - 62 y_21
      - 65 y_22 - 63 y_23 - 65 y_24 - 462 x_16 - 462 x_18 - 396 x_22
      + 66 c1a_11_int_slack_0 + 66 c1a_13_int_slack_0 + 66 c1a_15_int_slack_0
      + 66 c1a_16_int_slack_0 + 66 c1a_17_int_slack_0 + 132 c1a_17_int_slack_1
      + 66 c1a_18_int_slack_0 + 66 c1a_19_int_slack_0 + 66 c1a_21_int_slack_0
      + 66 c1a_21_int_slack_1 + 66 c1a_22_int_slack_0 + 66 c1a_23_int_slack_0
      + 66 c1a_23_int_slack_1 + [ 66 y_0^2 + 66 y_1^2 + 66 y_2^2 + 66 y_3^2
      + 66 y_4^2 + 66 y_5^2 + 66 y_6^2 + 66 y_7^2 + 66 y_8^2 + 66 y_9^2
      + 66 y_10^2 + 66 y_11^2 + 132 y_11*x_16 - 132 y_11*c1a_11_int_slack_0
      + 66 y_12^2 + 66 y_13^2 + 132 y_13*x_18 - 132 y_13*c1a_13_int_slack_0
      + 66 y_14^2 + 66 y_15^2 + 132 y_15*x_16 - 132 y_15*c1a_15_int_slack_0
      + 66 y_16^2 + 132 y_16*x_16 - 132 y_16*c1a_16_int_slack_0 + 66 y_17^2
      + 132 y_17*x_16 + 132 y_17*x_18 + 132 y_17*x_22
      - 132 y_17*c1a_17_int_slack_0 - 264 y_17*c1a_17_int_slack_1 + 66 y_18^2
      + 132 y_18*x_18 - 132 y_18*c1a_18_int_slack_0 + 66 y_19^2 + 132 y_19*x_18
      - 132 y_19*c1a_19_int_slack_0 + 66 y_20^2 + 66 y_21^2 + 132 y_21*x_16
      + 132 y_21*x_22 - 132 y_21*c1a_21_int_slack_0
      - 132 y_21*c1a_21_int_slack_1 + 66 y_22^2 + 132 y_22*x_22
      - 132 y_22*c1a_22_int_slack_0 + 66 y_23^2 + 132 y_23*x_18 + 132 y_23*x_22
      - 132 y_23*c1a_23_int_slack_0 - 132 y_23*c1a_23_int_slack_1 + 66 y_24^2
      + 396 x_16^2 + 264 x_16*x_18 + 396 x_16*x_22 - 132 x_16*c1a_11_int_slack_0
      - 132 x_16*c1a_15_int_slack_0 - 132 x_16*c1a_16_int_slack_0
      - 132 x_16*c1a_17_int_slack_0 - 264 x_16*c1a_17_int_slack_1
      - 132 x_16*c1a_21_int_slack_0 - 132 x_16*c1a_21_int_slack_1 + 396 x_18^2
      + 396 x_18*x_22 - 132 x_18*c1a_13_int_slack_0
      - 132 x_18*c1a_17_int_slack_0 - 264 x_18*c1a_17_int_slack_1
      - 132 x_18*c1a_18_int_slack_0 - 132 x_18*c1a_19_int_slack_0
      - 132 x_18*c1a_23_int_slack_0 - 132 x_18*c1a_23_int_slack_1 + 330 x_22^2
      - 132 x_22*c1a_17_int_slack_0 - 264 x_22*c1a_17_int_slack_1
      - 132 x_22*c1a_21_int_slack_0 - 132 x_22*c1a_21_int_slack_1
      - 132 x_22*c1a_22_int_slack_0 - 132 x_22*c1a_23_int_slack_0
      - 132 x_22*c1a_23_int_slack_1 + 66 c1a_11_int_slack_0^2
      + 66 c1a_13_int_slack_0^2 + 66 c1a_15_int_slack_0^2
      + 66 c1a_16_int_slack_0^2 + 66 c1a_17_int_slack_0^2
      + 264 c1a_17_int_slack_0*c1a_17_int_slack_1 + 264 c1a_17_int_slack_1^2
      + 66 c1a_18_int_slack_0^2 + 66 c1a_19_int_slack_0^2
      + 66 c1a_21_int_slack_0^2 + 132 c1a_21_int_slack_0*c1a_21_int_slack_1
      + 66 c1a_21_int_slack_1^2 + 66 c1a_22_int_slack_0^2
      + 66 c1a_23_int_slack_0^2 + 132 c1a_23_int_slack_0*c1a_23_int_slack_1
      + 66 c1a_23_int_slack_1^2 ]/2 + 957
Subject To

Bounds
 0 <= y_0 <= 1
 0 <= y_1 <= 1
 0 <= y_2 <= 1
 0 <= y_3 <= 1
 0 <= y_4 <= 1
 0 <= y_5 <= 1
 0 <= y_6 <= 1
 0 <= y_7 <= 1
 0 <= y_8 <= 1
 0 <= y_9 <= 1
 0 <= y_10 <= 1
 0 <= y_11 <= 1
 0 <= y_12 <= 1
 0 <= y_13 <= 1
 0 <= y_14 <= 1
 0 <= y_15 <= 1
 0 <= y_16 <= 1
 0 <= y_17 <= 1
 0 <= y_18 <= 1
 0 <= y_19 <= 1
 0 <= y_20 <= 1
 0 <= y_21 <= 1
 0 <= y_22 <= 1
 0 <= y_23 <= 1
 0 <= y_24 <= 1
 0 <= x_16 <= 1
 0 <= x_18 <= 1
 0 <= x_22 <= 1
 0 <= c1a_11_int_slack_0 <= 1
 0 <= c1a_13_int_slack_0 <= 1
 0 <= c1a_15_int_slack_0 <= 1
 0 <= c1a_16_int_slack_0 <= 1
 0 <= c1a_17_int_slack_0 <= 1
 0 <= c1a_17_int_slack_1 <= 1
 0 <= c1a_18_int_slack_0 <= 1
 0 <= c1a_19_int_slack_0 <= 1
 0 <= c1a_21_int_slack_0 <= 1
 0 <= c1a_21_int_slack_1 <= 1
 0 <= c1a_22_int_slack_0 <= 1
 0 <= c1a_23_int_slack_0 <= 1
 0 <= c1a_23_int_slack_1 <= 1

Binaries
 y_0 y_1 y_2 y_3 y_4 y_5 y_6 y_7 y_8 y_9 y_10 y_11 y_12 y_13 y_14 y_15 y_16 y_17
 y_18 y_19 y_20 y_21 y_22 y_23 y_24 x_16 x_18 x_22 c1a_11_int_slack_0
 c1a_13_int_slack_0 c1a_15_int_slack_0 c1a_16_int_slack_0 c1a_17_int_slack_0
 c1a_17_int_slack_1 c1a_18_int_slack_0 c1a_19_int_slack_0 c1a_21_int_slack_0
 c1a_21_int_slack_1 c1a_22_int_slack_0 c1a_23_int_slack_0 c1a_23_int_slack_1
End
