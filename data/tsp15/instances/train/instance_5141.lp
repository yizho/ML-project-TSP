\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: TSP

Minimize
 obj: 64.123991795808 x_0_1 + 45.347265850913 x_0_2 + 25.569618758798 x_0_3
      + 24.886791308559 x_0_4 + 20.935898448765 x_0_5 + 39.345876444132 x_0_6
      + 60.820749485439 x_0_7 + 59.900897334417 x_0_8 + 55.243812256364 x_0_9
      + 9.006306129515 x_0_10 + 47.792891652647 x_0_11 + 30.125539418654 x_0_12
      + 22.108642602952 x_0_13 + 38.927272782569 x_0_14 + 64.123991795808 x_1_0
      + 91.940639844733 x_1_2 + 89.508232535676 x_1_3 + 40.847749562213 x_1_4
      + 63.720534781638 x_1_5 + 96.203952034160 x_1_6 + 49.484892883470 x_1_7
      + 76.697880244675 x_1_8 + 14.430999479008 x_1_9 + 72.589190799053 x_1_10
      + 79.426521681768 x_1_11 + 83.250290236797 x_1_12 + 46.212457903179 x_1_13
      + 47.467118942054 x_1_14 + 45.347265850913 x_2_0 + 91.940639844733 x_2_1
      + 45.946823154747 x_2_3 + 54.245265121687 x_2_4 + 65.936191796244 x_2_5
      + 17.406885625804 x_2_6 + 59.678634303518 x_2_7 + 35.991581928724 x_2_8
      + 78.512166659928 x_2_9 + 45.948453045821 x_2_10 + 19.473490068478 x_2_11
      + 15.790613660627 x_2_12 + 64.351655236605 x_2_13 + 46.002835899506 x_2_14
      + 25.569618758798 x_3_0 + 89.508232535676 x_3_1 + 45.946823154747 x_3_2
      + 50.327876873515 x_3_4 + 33.704984007288 x_3_5 + 31.357324103362 x_3_6
      + 82.402904411721 x_3_7 + 73.361209958396 x_3_8 + 80.792489317406 x_3_9
      + 16.922931171014 x_3_10 + 57.614829269094 x_3_11 + 31.576630869933 x_3_12
      + 45.125196804285 x_3_13 + 60.903632496626 x_3_14 + 24.886791308559 x_4_0
      + 40.847749562213 x_4_1 + 54.245265121687 x_4_2 + 50.327876873515 x_4_3
      + 34.110671910849 x_4_5 + 55.854086725460 x_4_6 + 41.265296879127 x_4_7
      + 52.109747595485 x_4_8 + 30.555661235421 x_4_9 + 33.868433834745 x_4_10
      + 46.921059503690 x_4_11 + 43.265991688773 x_4_12 + 18.523108589262 x_4_13
      + 21.822986101012 x_4_14 + 20.935898448765 x_5_0 + 63.720534781638 x_5_1
      + 65.936191796244 x_5_2 + 33.704984007288 x_5_3 + 34.110671910849 x_5_4
      + 57.768431125761 x_5_6 + 75.076263880974 x_5_7 + 79.462033983757 x_5_8
      + 59.232867476934 x_5_9 + 21.186995089405 x_5_10 + 68.444457115438 x_5_11
      + 50.374349249495 x_5_12 + 18.180362860365 x_5_13 + 54.164744364421 x_5_14
      + 39.345876444132 x_6_0 + 96.203952034160 x_6_1 + 17.406885625804 x_6_2
      + 31.357324103362 x_6_3 + 55.854086725460 x_6_4 + 57.768431125761 x_6_5
      + 71.781727119645 x_6_7 + 52.125746636230 x_6_8 + 84.026272714164 x_6_9
      + 36.596723499711 x_6_10 + 35.096797885240 x_6_11 + 13.208431160870 x_6_12
      + 60.973408579093 x_6_13 + 54.353210158412 x_6_14 + 60.820749485439 x_7_0
      + 49.484892883470 x_7_1 + 59.678634303518 x_7_2 + 82.402904411721 x_7_3
      + 41.265296879127 x_7_4 + 75.076263880974 x_7_5 + 71.781727119645 x_7_6
      + 30.998036705177 x_7_8 + 35.878703251801 x_7_9 + 69.071860666629 x_7_10
      + 41.288740418898 x_7_11 + 59.498327244725 x_7_12 + 59.545090069046 x_7_13
      + 21.907083723434 x_7_14 + 59.900897334417 x_8_0 + 76.697880244675 x_8_1
      + 35.991581928724 x_8_2 + 73.361209958396 x_8_3 + 52.109747595485 x_8_4
      + 79.462033983757 x_8_5 + 52.125746636230 x_8_6 + 30.998036705177 x_8_7
      + 62.285427737864 x_8_9 + 65.481016019100 x_8_10 + 17.040465437918 x_8_11
      + 43.249024894975 x_8_12 + 69.308744468605 x_8_13 + 31.908391749861 x_8_14
      + 55.243812256364 x_9_0 + 14.430999479008 x_9_1 + 78.512166659928 x_9_2
      + 80.792489317406 x_9_3 + 30.555661235421 x_9_4 + 59.232867476934 x_9_5
      + 84.026272714164 x_9_6 + 35.878703251801 x_9_7 + 62.285427737864 x_9_8
      + 64.124573329845 x_9_10 + 65.249955056562 x_9_11 + 70.890980533318 x_9_12
      + 41.053252516373 x_9_13 + 33.322532784791 x_9_14 + 9.006306129515 x_10_0
      + 72.589190799053 x_10_1 + 45.948453045821 x_10_2 + 16.922931171014 x_10_3
      + 33.868433834745 x_10_4 + 21.186995089405 x_10_5 + 36.596723499711 x_10_6
      + 69.071860666629 x_10_7 + 65.481016019100 x_10_8 + 64.124573329845 x_10_9
      + 51.858543251858 x_10_11 + 30.163143888176 x_10_12
      + 28.653788074995 x_10_13 + 47.176914408133 x_10_14
      + 47.792891652647 x_11_0 + 79.426521681768 x_11_1 + 19.473490068478 x_11_2
      + 57.614829269094 x_11_3 + 46.921059503690 x_11_4 + 68.444457115438 x_11_5
      + 35.096797885240 x_11_6 + 41.288740418898 x_11_7 + 17.040465437918 x_11_8
      + 65.249955056562 x_11_9 + 51.858543251858 x_11_10
      + 26.640379189697 x_11_12 + 61.569570930989 x_11_13
      + 31.960115102217 x_11_14 + 30.125539418654 x_12_0
      + 83.250290236797 x_12_1 + 15.790613660627 x_12_2 + 31.576630869933 x_12_3
      + 43.265991688773 x_12_4 + 50.374349249495 x_12_5 + 13.208431160870 x_12_6
      + 59.498327244725 x_12_7 + 43.249024894975 x_12_8 + 70.890980533318 x_12_9
      + 30.163143888176 x_12_10 + 26.640379189697 x_12_11
      + 50.410674645518 x_12_13 + 41.287635717384 x_12_14
      + 22.108642602952 x_13_0 + 46.212457903179 x_13_1 + 64.351655236605 x_13_2
      + 45.125196804285 x_13_3 + 18.523108589262 x_13_4 + 18.180362860365 x_13_5
      + 60.973408579093 x_13_6 + 59.545090069046 x_13_7 + 69.308744468605 x_13_8
      + 41.053252516373 x_13_9 + 28.653788074995 x_13_10
      + 61.569570930989 x_13_11 + 50.410674645518 x_13_12
      + 40.243973317035 x_13_14 + 38.927272782569 x_14_0
      + 47.467118942054 x_14_1 + 46.002835899506 x_14_2 + 60.903632496626 x_14_3
      + 21.822986101012 x_14_4 + 54.164744364421 x_14_5 + 54.353210158412 x_14_6
      + 21.907083723434 x_14_7 + 31.908391749861 x_14_8 + 33.322532784791 x_14_9
      + 47.176914408133 x_14_10 + 31.960115102217 x_14_11
      + 41.287635717384 x_14_12 + 40.243973317035 x_14_13
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
