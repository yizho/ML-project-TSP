\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: TSP

Minimize
 obj: 57.212386592986 x_0_1 + 67.046579152069 x_0_2 + 49.105849768632 x_0_3
      + 38.632202253920 x_0_4 + 9.207494735853 x_0_5 + 6.409301488333 x_0_6
      + 29.464817925112 x_0_7 + 16.625634605642 x_0_8 + 22.117120908026 x_0_9
      + 50.910296771814 x_0_10 + 46.363599434485 x_0_11 + 29.952580045502 x_0_12
      + 37.966392771055 x_0_13 + 66.046884235836 x_0_14 + 57.212386592986 x_1_0
      + 32.523842045358 x_1_2 + 61.381449417000 x_1_3 + 42.412579264080 x_1_4
      + 58.065920407869 x_1_5 + 52.969211869351 x_1_6 + 52.576894967111 x_1_7
      + 73.304574318453 x_1_8 + 48.107010200740 x_1_9 + 80.400539833410 x_1_10
      + 93.996789678686 x_1_11 + 42.042414961322 x_1_12 + 30.859185349481 x_1_13
      + 9.361340904657 x_1_14 + 67.046579152069 x_2_0 + 32.523842045358 x_2_1
      + 89.113702754212 x_2_3 + 32.944829676006 x_2_4 + 63.211439746212 x_2_5
      + 60.949929802243 x_2_6 + 75.495195535189 x_2_7 + 79.587354828990 x_2_8
      + 68.400701643446 x_2_9 + 66.411076996682 x_2_10 + 111.578722834100 x_2_11
      + 39.259693374935 x_2_12 + 57.561996828262 x_2_13 + 36.557276632403 x_2_14
      + 49.105849768632 x_3_0 + 61.381449417000 x_3_1 + 89.113702754212 x_3_2
      + 75.358747787764 x_3_4 + 57.880130751573 x_3_5 + 52.247805700845 x_3_6
      + 19.834175502559 x_3_7 + 60.483148223311 x_3_8 + 28.319409382980 x_3_9
      + 99.266152646397 x_3_10 + 47.781189138018 x_3_11 + 67.651576092105 x_3_12
      + 31.603780357184 x_3_13 + 65.241065315747 x_3_14 + 38.632202253920 x_4_0
      + 42.412579264080 x_4_1 + 32.944829676006 x_4_2 + 75.358747787764 x_4_3
      + 32.497160870984 x_4_5 + 32.226486338865 x_4_6 + 57.244807833187 x_4_7
      + 48.215477430634 x_4_8 + 48.623218913485 x_4_9 + 38.148708400892 x_4_10
      + 84.981346363247 x_4_11 + 8.791705600123 x_4_12 + 48.124192692759 x_4_13
      + 51.229401543982 x_4_14 + 9.207494735853 x_5_0 + 58.065920407869 x_5_1
      + 63.211439746212 x_5_2 + 57.880130751573 x_5_3 + 32.497160870984 x_5_4
      + 6.472110491714 x_5_6 + 38.092585708545 x_5_7 + 16.442825555202 x_5_8
      + 30.181869941771 x_5_9 + 41.748377426290 x_5_10 + 53.674127790160 x_5_11
      + 24.364934652159 x_5_12 + 43.399957091918 x_5_13 + 67.259967466632 x_5_14
      + 6.409301488333 x_6_0 + 52.969211869351 x_6_1 + 60.949929802243 x_6_2
      + 52.247805700845 x_6_3 + 32.226486338865 x_6_4 + 6.472110491714 x_6_5
      + 32.413719731064 x_6_7 + 20.384780101038 x_6_8 + 24.191988102404 x_6_9
      + 47.029844666369 x_6_10 + 52.760041781159 x_6_11 + 23.543735763214 x_6_12
      + 36.952929679416 x_6_13 + 62.028969461510 x_6_14 + 29.464817925112 x_7_0
      + 52.576894967111 x_7_1 + 75.495195535189 x_7_2 + 19.834175502559 x_7_3
      + 57.244807833187 x_7_4 + 38.092585708545 x_7_5 + 32.413719731064 x_7_6
      + 42.219519763473 x_7_8 + 8.778572691642 x_7_9 + 79.435477417616 x_7_10
      + 42.733615225161 x_7_11 + 49.078938714684 x_7_12 + 22.297751039615 x_7_13
      + 58.853636255137 x_7_14 + 16.625634605642 x_8_0 + 73.304574318453 x_8_1
      + 79.587354828990 x_8_2 + 60.483148223311 x_8_3 + 48.215477430634 x_8_4
      + 16.442825555202 x_8_5 + 20.384780101038 x_8_6 + 42.219519763473 x_8_7
      + 36.612784527579 x_8_9 + 47.145034171531 x_8_10 + 42.553504097322 x_8_11
      + 40.510700009163 x_8_12 + 54.272560200932 x_8_13 + 82.297616316703 x_8_14
      + 22.117120908026 x_9_0 + 48.107010200740 x_9_1 + 68.400701643446 x_9_2
      + 28.319409382980 x_9_3 + 48.623218913485 x_9_4 + 30.181869941771 x_9_5
      + 24.191988102404 x_9_6 + 8.778572691642 x_9_7 + 36.612784527579 x_9_8
      + 71.067675125702 x_9_10 + 45.889801234833 x_9_11 + 40.368376425036 x_9_12
      + 20.242846892006 x_9_13 + 55.327043116452 x_9_14 + 50.910296771814 x_10_0
      + 80.400539833410 x_10_1 + 66.411076996682 x_10_2 + 99.266152646397 x_10_3
      + 38.148708400892 x_10_4 + 41.748377426290 x_10_5 + 47.029844666369 x_10_6
      + 79.435477417616 x_10_7 + 47.145034171531 x_10_8 + 71.067675125702 x_10_9
      + 89.579183962394 x_10_11 + 39.315714422276 x_10_12
      + 79.372838417671 x_10_13 + 89.340010987624 x_10_14
      + 46.363599434485 x_11_0 + 93.996789678686 x_11_1
      + 111.578722834100 x_11_2 + 47.781189138018 x_11_3
      + 84.981346363247 x_11_4 + 53.674127790160 x_11_5 + 52.760041781159 x_11_6
      + 42.733615225161 x_11_7 + 42.553504097322 x_11_8 + 45.889801234833 x_11_9
      + 89.579183962394 x_11_10 + 76.260576076153 x_11_12
      + 64.843686967329 x_11_13 + 101.045428100626 x_11_14
      + 29.952580045502 x_12_0 + 42.042414961322 x_12_1 + 39.259693374935 x_12_2
      + 67.651576092105 x_12_3 + 8.791705600123 x_12_4 + 24.364934652159 x_12_5
      + 23.543735763214 x_12_6 + 49.078938714684 x_12_7 + 40.510700009163 x_12_8
      + 40.368376425036 x_12_9 + 39.315714422276 x_12_10
      + 76.260576076153 x_12_11 + 42.052909780783 x_12_13
      + 51.293647759892 x_12_14 + 37.966392771055 x_13_0
      + 30.859185349481 x_13_1 + 57.561996828262 x_13_2 + 31.603780357184 x_13_3
      + 48.124192692759 x_13_4 + 43.399957091918 x_13_5 + 36.952929679416 x_13_6
      + 22.297751039615 x_13_7 + 54.272560200932 x_13_8 + 20.242846892006 x_13_9
      + 79.372838417671 x_13_10 + 64.843686967329 x_13_11
      + 42.052909780783 x_13_12 + 36.599311617695 x_13_14
      + 66.046884235836 x_14_0 + 9.361340904657 x_14_1 + 36.557276632403 x_14_2
      + 65.241065315747 x_14_3 + 51.229401543982 x_14_4 + 67.259967466632 x_14_5
      + 62.028969461510 x_14_6 + 58.853636255137 x_14_7 + 82.297616316703 x_14_8
      + 55.327043116452 x_14_9 + 89.340010987624 x_14_10
      + 101.045428100626 x_14_11 + 51.293647759892 x_14_12
      + 36.599311617695 x_14_13
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
