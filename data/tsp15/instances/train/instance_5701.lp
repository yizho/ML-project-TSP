\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: TSP

Minimize
 obj: 51.410931995738 x_0_1 + 60.400178380723 x_0_2 + 117.922249836893 x_0_3
      + 48.251232561571 x_0_4 + 14.874907752422 x_0_5 + 103.843276239254 x_0_6
      + 85.715086931113 x_0_7 + 79.293224479781 x_0_8 + 83.938931093533 x_0_9
      + 43.357296862648 x_0_10 + 76.291513208830 x_0_11 + 52.808594898346 x_0_12
      + 17.422209762370 x_0_13 + 83.428735515892 x_0_14 + 51.410931995738 x_1_0
      + 13.144642975980 x_1_2 + 79.978815294945 x_1_3 + 19.007863977261 x_1_4
      + 37.203316929551 x_1_5 + 65.468995715479 x_1_6 + 52.611657374501 x_1_7
      + 73.162822971588 x_1_8 + 70.920404577725 x_1_9 + 8.176264779100 x_1_10
      + 64.524549481452 x_1_11 + 28.422102861105 x_1_12 + 47.139669078303 x_1_13
      + 56.091431425204 x_1_14 + 60.400178380723 x_2_0 + 13.144642975980 x_2_1
      + 84.490880060825 x_2_3 + 31.779609257248 x_2_4 + 45.591582431268 x_2_5
      + 70.455136179628 x_2_6 + 60.444254633352 x_2_7 + 85.266658384361 x_2_8
      + 82.248078250498 x_2_9 + 19.928934748526 x_2_10 + 76.243778291902 x_2_11
      + 40.721622956253 x_2_12 + 58.678920495501 x_2_13 + 65.362613995182 x_2_14
      + 117.922249836893 x_3_0 + 79.978815294945 x_3_1 + 84.490880060825 x_3_2
      + 71.278824776240 x_3_4 + 108.474376571883 x_3_5 + 14.609741521999 x_3_6
      + 32.365920531728 x_3_7 + 64.298034365649 x_3_8 + 53.779121779019 x_3_9
      + 84.026861155280 x_3_10 + 56.918543150603 x_3_11 + 65.147728652892 x_3_12
      + 103.745669075225 x_3_13 + 37.183202038158 x_3_14 + 48.251232561571 x_4_0
      + 19.007863977261 x_4_1 + 31.779609257248 x_4_2 + 71.278824776240 x_4_3
      + 37.309946604672 x_4_5 + 56.797751959943 x_4_6 + 40.307323780857 x_4_7
      + 54.352349344236 x_4_8 + 52.825948062043 x_4_9 + 16.606092298571 x_4_10
      + 46.072596816339 x_4_11 + 9.849944044536 x_4_12 + 37.836369269512 x_4_13
      + 41.211093776382 x_4_14 + 14.874907752422 x_5_0 + 37.203316929551 x_5_1
      + 45.591582431268 x_5_2 + 108.474376571883 x_5_3 + 37.309946604672 x_5_4
      + 94.058275255527 x_5_6 + 76.900486452404 x_5_7 + 77.718567725319 x_5_8
      + 80.568215001684 x_5_9 + 29.376829407453 x_5_10 + 72.905147713733 x_5_11
      + 43.845045825594 x_5_12 + 21.004727550045 x_5_13 + 76.056315933580 x_5_14
      + 103.843276239254 x_6_0 + 65.468995715479 x_6_1 + 70.455136179628 x_6_2
      + 14.609741521999 x_6_3 + 56.797751959943 x_6_4 + 94.058275255527 x_6_5
      + 19.201759739347 x_6_7 + 56.882631367972 x_6_8 + 46.773582053209 x_6_9
      + 69.420359736056 x_6_10 + 48.152139674144 x_6_11 + 51.038624199787 x_6_12
      + 90.131739761800 x_6_13 + 26.124712454122 x_6_14 + 85.715086931113 x_7_0
      + 52.611657374501 x_7_1 + 60.444254633352 x_7_2 + 32.365920531728 x_7_3
      + 40.307323780857 x_7_4 + 76.900486452404 x_7_5 + 19.201759739347 x_7_6
      + 41.782623561299 x_7_8 + 32.938279408221 x_7_9 + 54.877736715811 x_7_10
      + 31.987115544640 x_7_11 + 33.141335322724 x_7_12 + 71.380954726945 x_7_13
      + 9.431125357050 x_7_14 + 79.293224479781 x_8_0 + 73.162822971588 x_8_1
      + 85.266658384361 x_8_2 + 64.298034365649 x_8_3 + 54.352349344236 x_8_4
      + 77.718567725319 x_8_5 + 56.882631367972 x_8_6 + 41.782623561299 x_8_7
      + 10.534171019907 x_8_9 + 70.475325636070 x_8_10 + 10.235096359671 x_8_11
      + 44.740726990803 x_8_12 + 61.887078746460 x_8_13 + 32.417015802241 x_8_14
      + 83.938931093533 x_9_0 + 70.920404577725 x_9_1 + 82.248078250498 x_9_2
      + 53.779121779019 x_9_3 + 52.825948062043 x_9_4 + 80.568215001684 x_9_5
      + 46.773582053209 x_9_6 + 32.938279408221 x_9_7 + 10.534171019907 x_9_8
      + 69.390405859846 x_9_10 + 7.688661390079 x_9_11 + 42.976144770395 x_9_12
      + 66.776033222128 x_9_13 + 23.524053311077 x_9_14 + 43.357296862648 x_10_0
      + 8.176264779100 x_10_1 + 19.928934748526 x_10_2 + 84.026861155280 x_10_3
      + 16.606092298571 x_10_4 + 29.376829407453 x_10_5 + 69.420359736056 x_10_6
      + 54.877736715811 x_10_7 + 70.475325636070 x_10_8 + 69.390405859846 x_10_9
      + 62.530839518285 x_10_11 + 26.438008602645 x_10_12
      + 39.133104802289 x_10_13 + 57.049864824283 x_10_14
      + 76.291513208830 x_11_0 + 64.524549481452 x_11_1 + 76.243778291902 x_11_2
      + 56.918543150603 x_11_3 + 46.072596816339 x_11_4 + 72.905147713733 x_11_5
      + 48.152139674144 x_11_6 + 31.987115544640 x_11_7 + 10.235096359671 x_11_8
      + 7.688661390079 x_11_9 + 62.530839518285 x_11_10
      + 36.262040337250 x_11_12 + 59.170850705418 x_11_13
      + 22.783088176364 x_11_14 + 52.808594898346 x_12_0
      + 28.422102861105 x_12_1 + 40.721622956253 x_12_2 + 65.147728652892 x_12_3
      + 9.849944044536 x_12_4 + 43.845045825594 x_12_5 + 51.038624199787 x_12_6
      + 33.141335322724 x_12_7 + 44.740726990803 x_12_8 + 42.976144770395 x_12_9
      + 26.438008602645 x_12_10 + 36.262040337250 x_12_11
      + 39.703473301399 x_12_13 + 32.555873418030 x_12_14
      + 17.422209762370 x_13_0 + 47.139669078303 x_13_1 + 58.678920495501 x_13_2
      + 103.745669075225 x_13_3 + 37.836369269512 x_13_4
      + 21.004727550045 x_13_5 + 90.131739761800 x_13_6 + 71.380954726945 x_13_7
      + 61.887078746460 x_13_8 + 66.776033222128 x_13_9
      + 39.133104802289 x_13_10 + 59.170850705418 x_13_11
      + 39.703473301399 x_13_12 + 68.034202259174 x_13_14
      + 83.428735515892 x_14_0 + 56.091431425204 x_14_1 + 65.362613995182 x_14_2
      + 37.183202038158 x_14_3 + 41.211093776382 x_14_4 + 76.056315933580 x_14_5
      + 26.124712454122 x_14_6 + 9.431125357050 x_14_7 + 32.417015802241 x_14_8
      + 23.524053311077 x_14_9 + 57.049864824283 x_14_10
      + 22.783088176364 x_14_11 + 32.555873418030 x_14_12
      + 68.034202259174 x_14_13
Subject To
 out_0: x_0_1 + x_0_2 + x_0_3 + x_0_4 + x_0_5 + x_0_6 + x_0_7 + x_0_8 + x_0_9
        + x_0_10 + x_0_11 + x_0_12 + x_0_13 + x_0_14 = 1
 in_0: x_1_0 + x_2_0 + x_3_0 + x_4_0 + x_5_0 + x_6_0 + x_7_0 + x_8_0 + x_9_0
       + x_10_0 + x_11_0 + x_12_0 + x_13_0 + x_14_0 = 1
 out_1: x_1_0 + x_1_2 + x_1_3 + x_1_4 + x_1_5 + x_1_6 + x_1_7 + x_1_8 + x_1_9
        + x_1_10 + x_1_11 + x_1_12 + x_1_13 + x_1_14 = 1
 in_1: x_0_1 + x_2_1 + x_3_1 + x_4_1 + x_5_1 + x_6_1 + x_7_1 + x_8_1 + x_9_1
       + x_10_1 + x_11_1 + x_12_1 + x_13_1 + x_14_1 = 1
 out_2: x_2_0 + x_2_1 + x_2_3 + x_2_4 + x_2_5 + x_2_6 + x_2_7 + x_2_8 + x_2_9
        + x_2_10 + x_2_11 + x_2_12 + x_2_13 + x_2_14 = 1
 in_2: x_0_2 + x_1_2 + x_3_2 + x_4_2 + x_5_2 + x_6_2 + x_7_2 + x_8_2 + x_9_2
       + x_10_2 + x_11_2 + x_12_2 + x_13_2 + x_14_2 = 1
 out_3: x_3_0 + x_3_1 + x_3_2 + x_3_4 + x_3_5 + x_3_6 + x_3_7 + x_3_8 + x_3_9
        + x_3_10 + x_3_11 + x_3_12 + x_3_13 + x_3_14 = 1
 in_3: x_0_3 + x_1_3 + x_2_3 + x_4_3 + x_5_3 + x_6_3 + x_7_3 + x_8_3 + x_9_3
       + x_10_3 + x_11_3 + x_12_3 + x_13_3 + x_14_3 = 1
 out_4: x_4_0 + x_4_1 + x_4_2 + x_4_3 + x_4_5 + x_4_6 + x_4_7 + x_4_8 + x_4_9
        + x_4_10 + x_4_11 + x_4_12 + x_4_13 + x_4_14 = 1
 in_4: x_0_4 + x_1_4 + x_2_4 + x_3_4 + x_5_4 + x_6_4 + x_7_4 + x_8_4 + x_9_4
       + x_10_4 + x_11_4 + x_12_4 + x_13_4 + x_14_4 = 1
 out_5: x_5_0 + x_5_1 + x_5_2 + x_5_3 + x_5_4 + x_5_6 + x_5_7 + x_5_8 + x_5_9
        + x_5_10 + x_5_11 + x_5_12 + x_5_13 + x_5_14 = 1
 in_5: x_0_5 + x_1_5 + x_2_5 + x_3_5 + x_4_5 + x_6_5 + x_7_5 + x_8_5 + x_9_5
       + x_10_5 + x_11_5 + x_12_5 + x_13_5 + x_14_5 = 1
 out_6: x_6_0 + x_6_1 + x_6_2 + x_6_3 + x_6_4 + x_6_5 + x_6_7 + x_6_8 + x_6_9
        + x_6_10 + x_6_11 + x_6_12 + x_6_13 + x_6_14 = 1
 in_6: x_0_6 + x_1_6 + x_2_6 + x_3_6 + x_4_6 + x_5_6 + x_7_6 + x_8_6 + x_9_6
       + x_10_6 + x_11_6 + x_12_6 + x_13_6 + x_14_6 = 1
 out_7: x_7_0 + x_7_1 + x_7_2 + x_7_3 + x_7_4 + x_7_5 + x_7_6 + x_7_8 + x_7_9
        + x_7_10 + x_7_11 + x_7_12 + x_7_13 + x_7_14 = 1
 in_7: x_0_7 + x_1_7 + x_2_7 + x_3_7 + x_4_7 + x_5_7 + x_6_7 + x_8_7 + x_9_7
       + x_10_7 + x_11_7 + x_12_7 + x_13_7 + x_14_7 = 1
 out_8: x_8_0 + x_8_1 + x_8_2 + x_8_3 + x_8_4 + x_8_5 + x_8_6 + x_8_7 + x_8_9
        + x_8_10 + x_8_11 + x_8_12 + x_8_13 + x_8_14 = 1
 in_8: x_0_8 + x_1_8 + x_2_8 + x_3_8 + x_4_8 + x_5_8 + x_6_8 + x_7_8 + x_9_8
       + x_10_8 + x_11_8 + x_12_8 + x_13_8 + x_14_8 = 1
 out_9: x_9_0 + x_9_1 + x_9_2 + x_9_3 + x_9_4 + x_9_5 + x_9_6 + x_9_7 + x_9_8
        + x_9_10 + x_9_11 + x_9_12 + x_9_13 + x_9_14 = 1
 in_9: x_0_9 + x_1_9 + x_2_9 + x_3_9 + x_4_9 + x_5_9 + x_6_9 + x_7_9 + x_8_9
       + x_10_9 + x_11_9 + x_12_9 + x_13_9 + x_14_9 = 1
 out_10: x_10_0 + x_10_1 + x_10_2 + x_10_3 + x_10_4 + x_10_5 + x_10_6 + x_10_7
         + x_10_8 + x_10_9 + x_10_11 + x_10_12 + x_10_13 + x_10_14 = 1
 in_10: x_0_10 + x_1_10 + x_2_10 + x_3_10 + x_4_10 + x_5_10 + x_6_10 + x_7_10
        + x_8_10 + x_9_10 + x_11_10 + x_12_10 + x_13_10 + x_14_10 = 1
 out_11: x_11_0 + x_11_1 + x_11_2 + x_11_3 + x_11_4 + x_11_5 + x_11_6 + x_11_7
         + x_11_8 + x_11_9 + x_11_10 + x_11_12 + x_11_13 + x_11_14 = 1
 in_11: x_0_11 + x_1_11 + x_2_11 + x_3_11 + x_4_11 + x_5_11 + x_6_11 + x_7_11
        + x_8_11 + x_9_11 + x_10_11 + x_12_11 + x_13_11 + x_14_11 = 1
 out_12: x_12_0 + x_12_1 + x_12_2 + x_12_3 + x_12_4 + x_12_5 + x_12_6 + x_12_7
         + x_12_8 + x_12_9 + x_12_10 + x_12_11 + x_12_13 + x_12_14 = 1
 in_12: x_0_12 + x_1_12 + x_2_12 + x_3_12 + x_4_12 + x_5_12 + x_6_12 + x_7_12
        + x_8_12 + x_9_12 + x_10_12 + x_11_12 + x_13_12 + x_14_12 = 1
 out_13: x_13_0 + x_13_1 + x_13_2 + x_13_3 + x_13_4 + x_13_5 + x_13_6 + x_13_7
         + x_13_8 + x_13_9 + x_13_10 + x_13_11 + x_13_12 + x_13_14 = 1
 in_13: x_0_13 + x_1_13 + x_2_13 + x_3_13 + x_4_13 + x_5_13 + x_6_13 + x_7_13
        + x_8_13 + x_9_13 + x_10_13 + x_11_13 + x_12_13 + x_14_13 = 1
 out_14: x_14_0 + x_14_1 + x_14_2 + x_14_3 + x_14_4 + x_14_5 + x_14_6 + x_14_7
         + x_14_8 + x_14_9 + x_14_10 + x_14_11 + x_14_12 + x_14_13 = 1
 in_14: x_0_14 + x_1_14 + x_2_14 + x_3_14 + x_4_14 + x_5_14 + x_6_14 + x_7_14
        + x_8_14 + x_9_14 + x_10_14 + x_11_14 + x_12_14 + x_13_14 = 1
 order_(0,_1): x_0_1 = 1 -> d_0 - d_1 = -1
 order_(0,_2): x_0_2 = 1 -> d_0 - d_2 = -1
 order_(0,_3): x_0_3 = 1 -> d_0 - d_3 = -1
 order_(0,_4): x_0_4 = 1 -> d_0 - d_4 = -1
 order_(0,_5): x_0_5 = 1 -> d_0 - d_5 = -1
 order_(0,_6): x_0_6 = 1 -> d_0 - d_6 = -1
 order_(0,_7): x_0_7 = 1 -> d_0 - d_7 = -1
 order_(0,_8): x_0_8 = 1 -> d_0 - d_8 = -1
 order_(0,_9): x_0_9 = 1 -> d_0 - d_9 = -1
 order_(0,_10): x_0_10 = 1 -> d_0 - d_10 = -1
 order_(0,_11): x_0_11 = 1 -> d_0 - d_11 = -1
 order_(0,_12): x_0_12 = 1 -> d_0 - d_12 = -1
 order_(0,_13): x_0_13 = 1 -> d_0 - d_13 = -1
 order_(0,_14): x_0_14 = 1 -> d_0 - d_14 = -1
 order_(1,_2): x_1_2 = 1 -> d_1 - d_2 = -1
 order_(1,_3): x_1_3 = 1 -> d_1 - d_3 = -1
 order_(1,_4): x_1_4 = 1 -> d_1 - d_4 = -1
 order_(1,_5): x_1_5 = 1 -> d_1 - d_5 = -1
 order_(1,_6): x_1_6 = 1 -> d_1 - d_6 = -1
 order_(1,_7): x_1_7 = 1 -> d_1 - d_7 = -1
 order_(1,_8): x_1_8 = 1 -> d_1 - d_8 = -1
 order_(1,_9): x_1_9 = 1 -> d_1 - d_9 = -1
 order_(1,_10): x_1_10 = 1 -> d_1 - d_10 = -1
 order_(1,_11): x_1_11 = 1 -> d_1 - d_11 = -1
 order_(1,_12): x_1_12 = 1 -> d_1 - d_12 = -1
 order_(1,_13): x_1_13 = 1 -> d_1 - d_13 = -1
 order_(1,_14): x_1_14 = 1 -> d_1 - d_14 = -1
 order_(2,_1): x_2_1 = 1 -> d_2 - d_1 = -1
 order_(2,_3): x_2_3 = 1 -> d_2 - d_3 = -1
 order_(2,_4): x_2_4 = 1 -> d_2 - d_4 = -1
 order_(2,_5): x_2_5 = 1 -> d_2 - d_5 = -1
 order_(2,_6): x_2_6 = 1 -> d_2 - d_6 = -1
 order_(2,_7): x_2_7 = 1 -> d_2 - d_7 = -1
 order_(2,_8): x_2_8 = 1 -> d_2 - d_8 = -1
 order_(2,_9): x_2_9 = 1 -> d_2 - d_9 = -1
 order_(2,_10): x_2_10 = 1 -> d_2 - d_10 = -1
 order_(2,_11): x_2_11 = 1 -> d_2 - d_11 = -1
 order_(2,_12): x_2_12 = 1 -> d_2 - d_12 = -1
 order_(2,_13): x_2_13 = 1 -> d_2 - d_13 = -1
 order_(2,_14): x_2_14 = 1 -> d_2 - d_14 = -1
 order_(3,_1): x_3_1 = 1 -> d_3 - d_1 = -1
 order_(3,_2): x_3_2 = 1 -> d_3 - d_2 = -1
 order_(3,_4): x_3_4 = 1 -> d_3 - d_4 = -1
 order_(3,_5): x_3_5 = 1 -> d_3 - d_5 = -1
 order_(3,_6): x_3_6 = 1 -> d_3 - d_6 = -1
 order_(3,_7): x_3_7 = 1 -> d_3 - d_7 = -1
 order_(3,_8): x_3_8 = 1 -> d_3 - d_8 = -1
 order_(3,_9): x_3_9 = 1 -> d_3 - d_9 = -1
 order_(3,_10): x_3_10 = 1 -> d_3 - d_10 = -1
 order_(3,_11): x_3_11 = 1 -> d_3 - d_11 = -1
 order_(3,_12): x_3_12 = 1 -> d_3 - d_12 = -1
 order_(3,_13): x_3_13 = 1 -> d_3 - d_13 = -1
 order_(3,_14): x_3_14 = 1 -> d_3 - d_14 = -1
 order_(4,_1): x_4_1 = 1 -> d_4 - d_1 = -1
 order_(4,_2): x_4_2 = 1 -> d_4 - d_2 = -1
 order_(4,_3): x_4_3 = 1 -> d_4 - d_3 = -1
 order_(4,_5): x_4_5 = 1 -> d_4 - d_5 = -1
 order_(4,_6): x_4_6 = 1 -> d_4 - d_6 = -1
 order_(4,_7): x_4_7 = 1 -> d_4 - d_7 = -1
 order_(4,_8): x_4_8 = 1 -> d_4 - d_8 = -1
 order_(4,_9): x_4_9 = 1 -> d_4 - d_9 = -1
 order_(4,_10): x_4_10 = 1 -> d_4 - d_10 = -1
 order_(4,_11): x_4_11 = 1 -> d_4 - d_11 = -1
 order_(4,_12): x_4_12 = 1 -> d_4 - d_12 = -1
 order_(4,_13): x_4_13 = 1 -> d_4 - d_13 = -1
 order_(4,_14): x_4_14 = 1 -> d_4 - d_14 = -1
 order_(5,_1): x_5_1 = 1 -> d_5 - d_1 = -1
 order_(5,_2): x_5_2 = 1 -> d_5 - d_2 = -1
 order_(5,_3): x_5_3 = 1 -> d_5 - d_3 = -1
 order_(5,_4): x_5_4 = 1 -> d_5 - d_4 = -1
 order_(5,_6): x_5_6 = 1 -> d_5 - d_6 = -1
 order_(5,_7): x_5_7 = 1 -> d_5 - d_7 = -1
 order_(5,_8): x_5_8 = 1 -> d_5 - d_8 = -1
 order_(5,_9): x_5_9 = 1 -> d_5 - d_9 = -1
 order_(5,_10): x_5_10 = 1 -> d_5 - d_10 = -1
 order_(5,_11): x_5_11 = 1 -> d_5 - d_11 = -1
 order_(5,_12): x_5_12 = 1 -> d_5 - d_12 = -1
 order_(5,_13): x_5_13 = 1 -> d_5 - d_13 = -1
 order_(5,_14): x_5_14 = 1 -> d_5 - d_14 = -1
 order_(6,_1): x_6_1 = 1 -> d_6 - d_1 = -1
 order_(6,_2): x_6_2 = 1 -> d_6 - d_2 = -1
 order_(6,_3): x_6_3 = 1 -> d_6 - d_3 = -1
 order_(6,_4): x_6_4 = 1 -> d_6 - d_4 = -1
 order_(6,_5): x_6_5 = 1 -> d_6 - d_5 = -1
 order_(6,_7): x_6_7 = 1 -> d_6 - d_7 = -1
 order_(6,_8): x_6_8 = 1 -> d_6 - d_8 = -1
 order_(6,_9): x_6_9 = 1 -> d_6 - d_9 = -1
 order_(6,_10): x_6_10 = 1 -> d_6 - d_10 = -1
 order_(6,_11): x_6_11 = 1 -> d_6 - d_11 = -1
 order_(6,_12): x_6_12 = 1 -> d_6 - d_12 = -1
 order_(6,_13): x_6_13 = 1 -> d_6 - d_13 = -1
 order_(6,_14): x_6_14 = 1 -> d_6 - d_14 = -1
 order_(7,_1): x_7_1 = 1 -> d_7 - d_1 = -1
 order_(7,_2): x_7_2 = 1 -> d_7 - d_2 = -1
 order_(7,_3): x_7_3 = 1 -> d_7 - d_3 = -1
 order_(7,_4): x_7_4 = 1 -> d_7 - d_4 = -1
 order_(7,_5): x_7_5 = 1 -> d_7 - d_5 = -1
 order_(7,_6): x_7_6 = 1 -> d_7 - d_6 = -1
 order_(7,_8): x_7_8 = 1 -> d_7 - d_8 = -1
 order_(7,_9): x_7_9 = 1 -> d_7 - d_9 = -1
 order_(7,_10): x_7_10 = 1 -> d_7 - d_10 = -1
 order_(7,_11): x_7_11 = 1 -> d_7 - d_11 = -1
 order_(7,_12): x_7_12 = 1 -> d_7 - d_12 = -1
 order_(7,_13): x_7_13 = 1 -> d_7 - d_13 = -1
 order_(7,_14): x_7_14 = 1 -> d_7 - d_14 = -1
 order_(8,_1): x_8_1 = 1 -> d_8 - d_1 = -1
 order_(8,_2): x_8_2 = 1 -> d_8 - d_2 = -1
 order_(8,_3): x_8_3 = 1 -> d_8 - d_3 = -1
 order_(8,_4): x_8_4 = 1 -> d_8 - d_4 = -1
 order_(8,_5): x_8_5 = 1 -> d_8 - d_5 = -1
 order_(8,_6): x_8_6 = 1 -> d_8 - d_6 = -1
 order_(8,_7): x_8_7 = 1 -> d_8 - d_7 = -1
 order_(8,_9): x_8_9 = 1 -> d_8 - d_9 = -1
 order_(8,_10): x_8_10 = 1 -> d_8 - d_10 = -1
 order_(8,_11): x_8_11 = 1 -> d_8 - d_11 = -1
 order_(8,_12): x_8_12 = 1 -> d_8 - d_12 = -1
 order_(8,_13): x_8_13 = 1 -> d_8 - d_13 = -1
 order_(8,_14): x_8_14 = 1 -> d_8 - d_14 = -1
 order_(9,_1): x_9_1 = 1 -> d_9 - d_1 = -1
 order_(9,_2): x_9_2 = 1 -> d_9 - d_2 = -1
 order_(9,_3): x_9_3 = 1 -> d_9 - d_3 = -1
 order_(9,_4): x_9_4 = 1 -> d_9 - d_4 = -1
 order_(9,_5): x_9_5 = 1 -> d_9 - d_5 = -1
 order_(9,_6): x_9_6 = 1 -> d_9 - d_6 = -1
 order_(9,_7): x_9_7 = 1 -> d_9 - d_7 = -1
 order_(9,_8): x_9_8 = 1 -> d_9 - d_8 = -1
 order_(9,_10): x_9_10 = 1 -> d_9 - d_10 = -1
 order_(9,_11): x_9_11 = 1 -> d_9 - d_11 = -1
 order_(9,_12): x_9_12 = 1 -> d_9 - d_12 = -1
 order_(9,_13): x_9_13 = 1 -> d_9 - d_13 = -1
 order_(9,_14): x_9_14 = 1 -> d_9 - d_14 = -1
 order_(10,_1): x_10_1 = 1 -> d_10 - d_1 = -1
 order_(10,_2): x_10_2 = 1 -> d_10 - d_2 = -1
 order_(10,_3): x_10_3 = 1 -> d_10 - d_3 = -1
 order_(10,_4): x_10_4 = 1 -> d_10 - d_4 = -1
 order_(10,_5): x_10_5 = 1 -> d_10 - d_5 = -1
 order_(10,_6): x_10_6 = 1 -> d_10 - d_6 = -1
 order_(10,_7): x_10_7 = 1 -> d_10 - d_7 = -1
 order_(10,_8): x_10_8 = 1 -> d_10 - d_8 = -1
 order_(10,_9): x_10_9 = 1 -> d_10 - d_9 = -1
 order_(10,_11): x_10_11 = 1 -> d_10 - d_11 = -1
 order_(10,_12): x_10_12 = 1 -> d_10 - d_12 = -1
 order_(10,_13): x_10_13 = 1 -> d_10 - d_13 = -1
 order_(10,_14): x_10_14 = 1 -> d_10 - d_14 = -1
 order_(11,_1): x_11_1 = 1 -> d_11 - d_1 = -1
 order_(11,_2): x_11_2 = 1 -> d_11 - d_2 = -1
 order_(11,_3): x_11_3 = 1 -> d_11 - d_3 = -1
 order_(11,_4): x_11_4 = 1 -> d_11 - d_4 = -1
 order_(11,_5): x_11_5 = 1 -> d_11 - d_5 = -1
 order_(11,_6): x_11_6 = 1 -> d_11 - d_6 = -1
 order_(11,_7): x_11_7 = 1 -> d_11 - d_7 = -1
 order_(11,_8): x_11_8 = 1 -> d_11 - d_8 = -1
 order_(11,_9): x_11_9 = 1 -> d_11 - d_9 = -1
 order_(11,_10): x_11_10 = 1 -> d_11 - d_10 = -1
 order_(11,_12): x_11_12 = 1 -> d_11 - d_12 = -1
 order_(11,_13): x_11_13 = 1 -> d_11 - d_13 = -1
 order_(11,_14): x_11_14 = 1 -> d_11 - d_14 = -1
 order_(12,_1): x_12_1 = 1 -> d_12 - d_1 = -1
 order_(12,_2): x_12_2 = 1 -> d_12 - d_2 = -1
 order_(12,_3): x_12_3 = 1 -> d_12 - d_3 = -1
 order_(12,_4): x_12_4 = 1 -> d_12 - d_4 = -1
 order_(12,_5): x_12_5 = 1 -> d_12 - d_5 = -1
 order_(12,_6): x_12_6 = 1 -> d_12 - d_6 = -1
 order_(12,_7): x_12_7 = 1 -> d_12 - d_7 = -1
 order_(12,_8): x_12_8 = 1 -> d_12 - d_8 = -1
 order_(12,_9): x_12_9 = 1 -> d_12 - d_9 = -1
 order_(12,_10): x_12_10 = 1 -> d_12 - d_10 = -1
 order_(12,_11): x_12_11 = 1 -> d_12 - d_11 = -1
 order_(12,_13): x_12_13 = 1 -> d_12 - d_13 = -1
 order_(12,_14): x_12_14 = 1 -> d_12 - d_14 = -1
 order_(13,_1): x_13_1 = 1 -> d_13 - d_1 = -1
 order_(13,_2): x_13_2 = 1 -> d_13 - d_2 = -1
 order_(13,_3): x_13_3 = 1 -> d_13 - d_3 = -1
 order_(13,_4): x_13_4 = 1 -> d_13 - d_4 = -1
 order_(13,_5): x_13_5 = 1 -> d_13 - d_5 = -1
 order_(13,_6): x_13_6 = 1 -> d_13 - d_6 = -1
 order_(13,_7): x_13_7 = 1 -> d_13 - d_7 = -1
 order_(13,_8): x_13_8 = 1 -> d_13 - d_8 = -1
 order_(13,_9): x_13_9 = 1 -> d_13 - d_9 = -1
 order_(13,_10): x_13_10 = 1 -> d_13 - d_10 = -1
 order_(13,_11): x_13_11 = 1 -> d_13 - d_11 = -1
 order_(13,_12): x_13_12 = 1 -> d_13 - d_12 = -1
 order_(13,_14): x_13_14 = 1 -> d_13 - d_14 = -1
 order_(14,_1): x_14_1 = 1 -> d_14 - d_1 = -1
 order_(14,_2): x_14_2 = 1 -> d_14 - d_2 = -1
 order_(14,_3): x_14_3 = 1 -> d_14 - d_3 = -1
 order_(14,_4): x_14_4 = 1 -> d_14 - d_4 = -1
 order_(14,_5): x_14_5 = 1 -> d_14 - d_5 = -1
 order_(14,_6): x_14_6 = 1 -> d_14 - d_6 = -1
 order_(14,_7): x_14_7 = 1 -> d_14 - d_7 = -1
 order_(14,_8): x_14_8 = 1 -> d_14 - d_8 = -1
 order_(14,_9): x_14_9 = 1 -> d_14 - d_9 = -1
 order_(14,_10): x_14_10 = 1 -> d_14 - d_10 = -1
 order_(14,_11): x_14_11 = 1 -> d_14 - d_11 = -1
 order_(14,_12): x_14_12 = 1 -> d_14 - d_12 = -1
 order_(14,_13): x_14_13 = 1 -> d_14 - d_13 = -1

Bounds
 0 <= x_0_1 <= 1
 0 <= x_0_2 <= 1
 0 <= x_0_3 <= 1
 0 <= x_0_4 <= 1
 0 <= x_0_5 <= 1
 0 <= x_0_6 <= 1
 0 <= x_0_7 <= 1
 0 <= x_0_8 <= 1
 0 <= x_0_9 <= 1
 0 <= x_0_10 <= 1
 0 <= x_0_11 <= 1
 0 <= x_0_12 <= 1
 0 <= x_0_13 <= 1
 0 <= x_0_14 <= 1
 0 <= x_1_0 <= 1
 0 <= x_1_2 <= 1
 0 <= x_1_3 <= 1
 0 <= x_1_4 <= 1
 0 <= x_1_5 <= 1
 0 <= x_1_6 <= 1
 0 <= x_1_7 <= 1
 0 <= x_1_8 <= 1
 0 <= x_1_9 <= 1
 0 <= x_1_10 <= 1
 0 <= x_1_11 <= 1
 0 <= x_1_12 <= 1
 0 <= x_1_13 <= 1
 0 <= x_1_14 <= 1
 0 <= x_2_0 <= 1
 0 <= x_2_1 <= 1
 0 <= x_2_3 <= 1
 0 <= x_2_4 <= 1
 0 <= x_2_5 <= 1
 0 <= x_2_6 <= 1
 0 <= x_2_7 <= 1
 0 <= x_2_8 <= 1
 0 <= x_2_9 <= 1
 0 <= x_2_10 <= 1
 0 <= x_2_11 <= 1
 0 <= x_2_12 <= 1
 0 <= x_2_13 <= 1
 0 <= x_2_14 <= 1
 0 <= x_3_0 <= 1
 0 <= x_3_1 <= 1
 0 <= x_3_2 <= 1
 0 <= x_3_4 <= 1
 0 <= x_3_5 <= 1
 0 <= x_3_6 <= 1
 0 <= x_3_7 <= 1
 0 <= x_3_8 <= 1
 0 <= x_3_9 <= 1
 0 <= x_3_10 <= 1
 0 <= x_3_11 <= 1
 0 <= x_3_12 <= 1
 0 <= x_3_13 <= 1
 0 <= x_3_14 <= 1
 0 <= x_4_0 <= 1
 0 <= x_4_1 <= 1
 0 <= x_4_2 <= 1
 0 <= x_4_3 <= 1
 0 <= x_4_5 <= 1
 0 <= x_4_6 <= 1
 0 <= x_4_7 <= 1
 0 <= x_4_8 <= 1
 0 <= x_4_9 <= 1
 0 <= x_4_10 <= 1
 0 <= x_4_11 <= 1
 0 <= x_4_12 <= 1
 0 <= x_4_13 <= 1
 0 <= x_4_14 <= 1
 0 <= x_5_0 <= 1
 0 <= x_5_1 <= 1
 0 <= x_5_2 <= 1
 0 <= x_5_3 <= 1
 0 <= x_5_4 <= 1
 0 <= x_5_6 <= 1
 0 <= x_5_7 <= 1
 0 <= x_5_8 <= 1
 0 <= x_5_9 <= 1
 0 <= x_5_10 <= 1
 0 <= x_5_11 <= 1
 0 <= x_5_12 <= 1
 0 <= x_5_13 <= 1
 0 <= x_5_14 <= 1
 0 <= x_6_0 <= 1
 0 <= x_6_1 <= 1
 0 <= x_6_2 <= 1
 0 <= x_6_3 <= 1
 0 <= x_6_4 <= 1
 0 <= x_6_5 <= 1
 0 <= x_6_7 <= 1
 0 <= x_6_8 <= 1
 0 <= x_6_9 <= 1
 0 <= x_6_10 <= 1
 0 <= x_6_11 <= 1
 0 <= x_6_12 <= 1
 0 <= x_6_13 <= 1
 0 <= x_6_14 <= 1
 0 <= x_7_0 <= 1
 0 <= x_7_1 <= 1
 0 <= x_7_2 <= 1
 0 <= x_7_3 <= 1
 0 <= x_7_4 <= 1
 0 <= x_7_5 <= 1
 0 <= x_7_6 <= 1
 0 <= x_7_8 <= 1
 0 <= x_7_9 <= 1
 0 <= x_7_10 <= 1
 0 <= x_7_11 <= 1
 0 <= x_7_12 <= 1
 0 <= x_7_13 <= 1
 0 <= x_7_14 <= 1
 0 <= x_8_0 <= 1
 0 <= x_8_1 <= 1
 0 <= x_8_2 <= 1
 0 <= x_8_3 <= 1
 0 <= x_8_4 <= 1
 0 <= x_8_5 <= 1
 0 <= x_8_6 <= 1
 0 <= x_8_7 <= 1
 0 <= x_8_9 <= 1
 0 <= x_8_10 <= 1
 0 <= x_8_11 <= 1
 0 <= x_8_12 <= 1
 0 <= x_8_13 <= 1
 0 <= x_8_14 <= 1
 0 <= x_9_0 <= 1
 0 <= x_9_1 <= 1
 0 <= x_9_2 <= 1
 0 <= x_9_3 <= 1
 0 <= x_9_4 <= 1
 0 <= x_9_5 <= 1
 0 <= x_9_6 <= 1
 0 <= x_9_7 <= 1
 0 <= x_9_8 <= 1
 0 <= x_9_10 <= 1
 0 <= x_9_11 <= 1
 0 <= x_9_12 <= 1
 0 <= x_9_13 <= 1
 0 <= x_9_14 <= 1
 0 <= x_10_0 <= 1
 0 <= x_10_1 <= 1
 0 <= x_10_2 <= 1
 0 <= x_10_3 <= 1
 0 <= x_10_4 <= 1
 0 <= x_10_5 <= 1
 0 <= x_10_6 <= 1
 0 <= x_10_7 <= 1
 0 <= x_10_8 <= 1
 0 <= x_10_9 <= 1
 0 <= x_10_11 <= 1
 0 <= x_10_12 <= 1
 0 <= x_10_13 <= 1
 0 <= x_10_14 <= 1
 0 <= x_11_0 <= 1
 0 <= x_11_1 <= 1
 0 <= x_11_2 <= 1
 0 <= x_11_3 <= 1
 0 <= x_11_4 <= 1
 0 <= x_11_5 <= 1
 0 <= x_11_6 <= 1
 0 <= x_11_7 <= 1
 0 <= x_11_8 <= 1
 0 <= x_11_9 <= 1
 0 <= x_11_10 <= 1
 0 <= x_11_12 <= 1
 0 <= x_11_13 <= 1
 0 <= x_11_14 <= 1
 0 <= x_12_0 <= 1
 0 <= x_12_1 <= 1
 0 <= x_12_2 <= 1
 0 <= x_12_3 <= 1
 0 <= x_12_4 <= 1
 0 <= x_12_5 <= 1
 0 <= x_12_6 <= 1
 0 <= x_12_7 <= 1
 0 <= x_12_8 <= 1
 0 <= x_12_9 <= 1
 0 <= x_12_10 <= 1
 0 <= x_12_11 <= 1
 0 <= x_12_13 <= 1
 0 <= x_12_14 <= 1
 0 <= x_13_0 <= 1
 0 <= x_13_1 <= 1
 0 <= x_13_2 <= 1
 0 <= x_13_3 <= 1
 0 <= x_13_4 <= 1
 0 <= x_13_5 <= 1
 0 <= x_13_6 <= 1
 0 <= x_13_7 <= 1
 0 <= x_13_8 <= 1
 0 <= x_13_9 <= 1
 0 <= x_13_10 <= 1
 0 <= x_13_11 <= 1
 0 <= x_13_12 <= 1
 0 <= x_13_14 <= 1
 0 <= x_14_0 <= 1
 0 <= x_14_1 <= 1
 0 <= x_14_2 <= 1
 0 <= x_14_3 <= 1
 0 <= x_14_4 <= 1
 0 <= x_14_5 <= 1
 0 <= x_14_6 <= 1
 0 <= x_14_7 <= 1
 0 <= x_14_8 <= 1
 0 <= x_14_9 <= 1
 0 <= x_14_10 <= 1
 0 <= x_14_11 <= 1
 0 <= x_14_12 <= 1
 0 <= x_14_13 <= 1

Binaries
 x_0_1 x_0_2 x_0_3 x_0_4 x_0_5 x_0_6 x_0_7 x_0_8 x_0_9 x_0_10 x_0_11 x_0_12
 x_0_13 x_0_14 x_1_0 x_1_2 x_1_3 x_1_4 x_1_5 x_1_6 x_1_7 x_1_8 x_1_9 x_1_10
 x_1_11 x_1_12 x_1_13 x_1_14 x_2_0 x_2_1 x_2_3 x_2_4 x_2_5 x_2_6 x_2_7 x_2_8
 x_2_9 x_2_10 x_2_11 x_2_12 x_2_13 x_2_14 x_3_0 x_3_1 x_3_2 x_3_4 x_3_5 x_3_6
 x_3_7 x_3_8 x_3_9 x_3_10 x_3_11 x_3_12 x_3_13 x_3_14 x_4_0 x_4_1 x_4_2 x_4_3
 x_4_5 x_4_6 x_4_7 x_4_8 x_4_9 x_4_10 x_4_11 x_4_12 x_4_13 x_4_14 x_5_0 x_5_1
 x_5_2 x_5_3 x_5_4 x_5_6 x_5_7 x_5_8 x_5_9 x_5_10 x_5_11 x_5_12 x_5_13 x_5_14
 x_6_0 x_6_1 x_6_2 x_6_3 x_6_4 x_6_5 x_6_7 x_6_8 x_6_9 x_6_10 x_6_11 x_6_12
 x_6_13 x_6_14 x_7_0 x_7_1 x_7_2 x_7_3 x_7_4 x_7_5 x_7_6 x_7_8 x_7_9 x_7_10
 x_7_11 x_7_12 x_7_13 x_7_14 x_8_0 x_8_1 x_8_2 x_8_3 x_8_4 x_8_5 x_8_6 x_8_7
 x_8_9 x_8_10 x_8_11 x_8_12 x_8_13 x_8_14 x_9_0 x_9_1 x_9_2 x_9_3 x_9_4 x_9_5
 x_9_6 x_9_7 x_9_8 x_9_10 x_9_11 x_9_12 x_9_13 x_9_14 x_10_0 x_10_1 x_10_2
 x_10_3 x_10_4 x_10_5 x_10_6 x_10_7 x_10_8 x_10_9 x_10_11 x_10_12 x_10_13
 x_10_14 x_11_0 x_11_1 x_11_2 x_11_3 x_11_4 x_11_5 x_11_6 x_11_7 x_11_8 x_11_9
 x_11_10 x_11_12 x_11_13 x_11_14 x_12_0 x_12_1 x_12_2 x_12_3 x_12_4 x_12_5
 x_12_6 x_12_7 x_12_8 x_12_9 x_12_10 x_12_11 x_12_13 x_12_14 x_13_0 x_13_1
 x_13_2 x_13_3 x_13_4 x_13_5 x_13_6 x_13_7 x_13_8 x_13_9 x_13_10 x_13_11 x_13_12
 x_13_14 x_14_0 x_14_1 x_14_2 x_14_3 x_14_4 x_14_5 x_14_6 x_14_7 x_14_8 x_14_9
 x_14_10 x_14_11 x_14_12 x_14_13
End
