\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: TSP

Minimize
 obj: 54.543979108281 x_0_1 + 30.518831547447 x_0_2 + 65.235600364338 x_0_3
      + 62.383360173966 x_0_4 + 33.809953854793 x_0_5 + 37.239338764028 x_0_6
      + 18.834981083165 x_0_7 + 31.859345422043 x_0_8 + 76.781528086659 x_0_9
      + 54.543979108281 x_1_0 + 69.811412635247 x_1_2 + 24.608995986186 x_1_3
      + 73.975713257167 x_1_4 + 87.972938003770 x_1_5 + 57.119775265892 x_1_6
      + 39.478208994604 x_1_7 + 43.136927507933 x_1_8 + 79.852255324863 x_1_9
      + 30.518831547447 x_2_0 + 69.811412635247 x_2_1 + 87.493485838653 x_2_3
      + 38.516438023850 x_2_4 + 33.327634840273 x_2_5 + 20.074575315942 x_2_6
      + 46.168942613795 x_2_7 + 61.344706959802 x_2_8 + 54.324892231512 x_2_9
      + 65.235600364338 x_3_0 + 24.608995986186 x_3_1 + 87.493485838653 x_3_2
      + 97.486130283676 x_3_4 + 98.674646392814 x_3_5 + 78.271308053340 x_3_6
      + 46.728053980204 x_3_7 + 41.235631822615 x_3_8 + 104.238995823382 x_3_9
      + 62.383360173966 x_4_0 + 73.975713257167 x_4_1 + 38.516438023850 x_4_2
      + 97.486130283676 x_4_3 + 71.406538519293 x_4_5 + 25.354372572292 x_4_6
      + 70.689722336507 x_4_7 + 86.502691664806 x_4_8 + 15.856168435447 x_4_9
      + 33.809953854793 x_5_0 + 87.972938003770 x_5_1 + 33.327634840273 x_5_2
      + 98.674646392814 x_5_3 + 71.406538519293 x_5_4 + 52.327954493624 x_5_6
      + 51.946644677660 x_5_7 + 61.460932799629 x_5_8 + 87.261992341350 x_5_9
      + 37.239338764028 x_6_0 + 57.119775265892 x_6_1 + 20.074575315942 x_6_2
      + 78.271308053340 x_6_3 + 25.354372572292 x_6_4 + 52.327954493624 x_6_5
      + 45.811862091761 x_6_7 + 61.756589506862 x_6_8 + 39.571272569620 x_6_9
      + 18.834981083165 x_7_0 + 39.478208994604 x_7_1 + 46.168942613795 x_7_2
      + 46.728053980204 x_7_3 + 70.689722336507 x_7_4 + 51.946644677660 x_7_5
      + 45.811862091761 x_7_6 + 15.975419324474 x_7_8 + 83.108484937630 x_7_9
      + 31.859345422043 x_8_0 + 43.136927507933 x_8_1 + 61.344706959802 x_8_2
      + 41.235631822615 x_8_3 + 86.502691664806 x_8_4 + 61.460932799629 x_8_5
      + 61.756589506862 x_8_6 + 15.975419324474 x_8_7 + 98.535422985620 x_8_9
      + 76.781528086659 x_9_0 + 79.852255324863 x_9_1 + 54.324892231512 x_9_2
      + 104.238995823382 x_9_3 + 15.856168435447 x_9_4 + 87.261992341350 x_9_5
      + 39.571272569620 x_9_6 + 83.108484937630 x_9_7 + 98.535422985620 x_9_8
Subject To
 out_0: x_0_1 + x_0_2 + x_0_3 + x_0_4 + x_0_5 + x_0_6 + x_0_7 + x_0_8 + x_0_9
         = 1
 in_0: x_1_0 + x_2_0 + x_3_0 + x_4_0 + x_5_0 + x_6_0 + x_7_0 + x_8_0 + x_9_0 = 
       1
 out_1: x_1_0 + x_1_2 + x_1_3 + x_1_4 + x_1_5 + x_1_6 + x_1_7 + x_1_8 + x_1_9
         = 1
 in_1: x_0_1 + x_2_1 + x_3_1 + x_4_1 + x_5_1 + x_6_1 + x_7_1 + x_8_1 + x_9_1 = 
       1
 out_2: x_2_0 + x_2_1 + x_2_3 + x_2_4 + x_2_5 + x_2_6 + x_2_7 + x_2_8 + x_2_9
         = 1
 in_2: x_0_2 + x_1_2 + x_3_2 + x_4_2 + x_5_2 + x_6_2 + x_7_2 + x_8_2 + x_9_2 = 
       1
 out_3: x_3_0 + x_3_1 + x_3_2 + x_3_4 + x_3_5 + x_3_6 + x_3_7 + x_3_8 + x_3_9
         = 1
 in_3: x_0_3 + x_1_3 + x_2_3 + x_4_3 + x_5_3 + x_6_3 + x_7_3 + x_8_3 + x_9_3 = 
       1
 out_4: x_4_0 + x_4_1 + x_4_2 + x_4_3 + x_4_5 + x_4_6 + x_4_7 + x_4_8 + x_4_9
         = 1
 in_4: x_0_4 + x_1_4 + x_2_4 + x_3_4 + x_5_4 + x_6_4 + x_7_4 + x_8_4 + x_9_4 = 
       1
 out_5: x_5_0 + x_5_1 + x_5_2 + x_5_3 + x_5_4 + x_5_6 + x_5_7 + x_5_8 + x_5_9
         = 1
 in_5: x_0_5 + x_1_5 + x_2_5 + x_3_5 + x_4_5 + x_6_5 + x_7_5 + x_8_5 + x_9_5 = 
       1
 out_6: x_6_0 + x_6_1 + x_6_2 + x_6_3 + x_6_4 + x_6_5 + x_6_7 + x_6_8 + x_6_9
         = 1
 in_6: x_0_6 + x_1_6 + x_2_6 + x_3_6 + x_4_6 + x_5_6 + x_7_6 + x_8_6 + x_9_6 = 
       1
 out_7: x_7_0 + x_7_1 + x_7_2 + x_7_3 + x_7_4 + x_7_5 + x_7_6 + x_7_8 + x_7_9
         = 1
 in_7: x_0_7 + x_1_7 + x_2_7 + x_3_7 + x_4_7 + x_5_7 + x_6_7 + x_8_7 + x_9_7 = 
       1
 out_8: x_8_0 + x_8_1 + x_8_2 + x_8_3 + x_8_4 + x_8_5 + x_8_6 + x_8_7 + x_8_9
         = 1
 in_8: x_0_8 + x_1_8 + x_2_8 + x_3_8 + x_4_8 + x_5_8 + x_6_8 + x_7_8 + x_9_8 = 
       1
 out_9: x_9_0 + x_9_1 + x_9_2 + x_9_3 + x_9_4 + x_9_5 + x_9_6 + x_9_7 + x_9_8
         = 1
 in_9: x_0_9 + x_1_9 + x_2_9 + x_3_9 + x_4_9 + x_5_9 + x_6_9 + x_7_9 + x_8_9 = 
       1
 order_(0,_1): x_0_1 = 1 -> d_0 - d_1 = -1
 order_(0,_2): x_0_2 = 1 -> d_0 - d_2 = -1
 order_(0,_3): x_0_3 = 1 -> d_0 - d_3 = -1
 order_(0,_4): x_0_4 = 1 -> d_0 - d_4 = -1
 order_(0,_5): x_0_5 = 1 -> d_0 - d_5 = -1
 order_(0,_6): x_0_6 = 1 -> d_0 - d_6 = -1
 order_(0,_7): x_0_7 = 1 -> d_0 - d_7 = -1
 order_(0,_8): x_0_8 = 1 -> d_0 - d_8 = -1
 order_(0,_9): x_0_9 = 1 -> d_0 - d_9 = -1
 order_(1,_2): x_1_2 = 1 -> d_1 - d_2 = -1
 order_(1,_3): x_1_3 = 1 -> d_1 - d_3 = -1
 order_(1,_4): x_1_4 = 1 -> d_1 - d_4 = -1
 order_(1,_5): x_1_5 = 1 -> d_1 - d_5 = -1
 order_(1,_6): x_1_6 = 1 -> d_1 - d_6 = -1
 order_(1,_7): x_1_7 = 1 -> d_1 - d_7 = -1
 order_(1,_8): x_1_8 = 1 -> d_1 - d_8 = -1
 order_(1,_9): x_1_9 = 1 -> d_1 - d_9 = -1
 order_(2,_1): x_2_1 = 1 -> d_2 - d_1 = -1
 order_(2,_3): x_2_3 = 1 -> d_2 - d_3 = -1
 order_(2,_4): x_2_4 = 1 -> d_2 - d_4 = -1
 order_(2,_5): x_2_5 = 1 -> d_2 - d_5 = -1
 order_(2,_6): x_2_6 = 1 -> d_2 - d_6 = -1
 order_(2,_7): x_2_7 = 1 -> d_2 - d_7 = -1
 order_(2,_8): x_2_8 = 1 -> d_2 - d_8 = -1
 order_(2,_9): x_2_9 = 1 -> d_2 - d_9 = -1
 order_(3,_1): x_3_1 = 1 -> d_3 - d_1 = -1
 order_(3,_2): x_3_2 = 1 -> d_3 - d_2 = -1
 order_(3,_4): x_3_4 = 1 -> d_3 - d_4 = -1
 order_(3,_5): x_3_5 = 1 -> d_3 - d_5 = -1
 order_(3,_6): x_3_6 = 1 -> d_3 - d_6 = -1
 order_(3,_7): x_3_7 = 1 -> d_3 - d_7 = -1
 order_(3,_8): x_3_8 = 1 -> d_3 - d_8 = -1
 order_(3,_9): x_3_9 = 1 -> d_3 - d_9 = -1
 order_(4,_1): x_4_1 = 1 -> d_4 - d_1 = -1
 order_(4,_2): x_4_2 = 1 -> d_4 - d_2 = -1
 order_(4,_3): x_4_3 = 1 -> d_4 - d_3 = -1
 order_(4,_5): x_4_5 = 1 -> d_4 - d_5 = -1
 order_(4,_6): x_4_6 = 1 -> d_4 - d_6 = -1
 order_(4,_7): x_4_7 = 1 -> d_4 - d_7 = -1
 order_(4,_8): x_4_8 = 1 -> d_4 - d_8 = -1
 order_(4,_9): x_4_9 = 1 -> d_4 - d_9 = -1
 order_(5,_1): x_5_1 = 1 -> d_5 - d_1 = -1
 order_(5,_2): x_5_2 = 1 -> d_5 - d_2 = -1
 order_(5,_3): x_5_3 = 1 -> d_5 - d_3 = -1
 order_(5,_4): x_5_4 = 1 -> d_5 - d_4 = -1
 order_(5,_6): x_5_6 = 1 -> d_5 - d_6 = -1
 order_(5,_7): x_5_7 = 1 -> d_5 - d_7 = -1
 order_(5,_8): x_5_8 = 1 -> d_5 - d_8 = -1
 order_(5,_9): x_5_9 = 1 -> d_5 - d_9 = -1
 order_(6,_1): x_6_1 = 1 -> d_6 - d_1 = -1
 order_(6,_2): x_6_2 = 1 -> d_6 - d_2 = -1
 order_(6,_3): x_6_3 = 1 -> d_6 - d_3 = -1
 order_(6,_4): x_6_4 = 1 -> d_6 - d_4 = -1
 order_(6,_5): x_6_5 = 1 -> d_6 - d_5 = -1
 order_(6,_7): x_6_7 = 1 -> d_6 - d_7 = -1
 order_(6,_8): x_6_8 = 1 -> d_6 - d_8 = -1
 order_(6,_9): x_6_9 = 1 -> d_6 - d_9 = -1
 order_(7,_1): x_7_1 = 1 -> d_7 - d_1 = -1
 order_(7,_2): x_7_2 = 1 -> d_7 - d_2 = -1
 order_(7,_3): x_7_3 = 1 -> d_7 - d_3 = -1
 order_(7,_4): x_7_4 = 1 -> d_7 - d_4 = -1
 order_(7,_5): x_7_5 = 1 -> d_7 - d_5 = -1
 order_(7,_6): x_7_6 = 1 -> d_7 - d_6 = -1
 order_(7,_8): x_7_8 = 1 -> d_7 - d_8 = -1
 order_(7,_9): x_7_9 = 1 -> d_7 - d_9 = -1
 order_(8,_1): x_8_1 = 1 -> d_8 - d_1 = -1
 order_(8,_2): x_8_2 = 1 -> d_8 - d_2 = -1
 order_(8,_3): x_8_3 = 1 -> d_8 - d_3 = -1
 order_(8,_4): x_8_4 = 1 -> d_8 - d_4 = -1
 order_(8,_5): x_8_5 = 1 -> d_8 - d_5 = -1
 order_(8,_6): x_8_6 = 1 -> d_8 - d_6 = -1
 order_(8,_7): x_8_7 = 1 -> d_8 - d_7 = -1
 order_(8,_9): x_8_9 = 1 -> d_8 - d_9 = -1
 order_(9,_1): x_9_1 = 1 -> d_9 - d_1 = -1
 order_(9,_2): x_9_2 = 1 -> d_9 - d_2 = -1
 order_(9,_3): x_9_3 = 1 -> d_9 - d_3 = -1
 order_(9,_4): x_9_4 = 1 -> d_9 - d_4 = -1
 order_(9,_5): x_9_5 = 1 -> d_9 - d_5 = -1
 order_(9,_6): x_9_6 = 1 -> d_9 - d_6 = -1
 order_(9,_7): x_9_7 = 1 -> d_9 - d_7 = -1
 order_(9,_8): x_9_8 = 1 -> d_9 - d_8 = -1

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
 0 <= x_1_0 <= 1
 0 <= x_1_2 <= 1
 0 <= x_1_3 <= 1
 0 <= x_1_4 <= 1
 0 <= x_1_5 <= 1
 0 <= x_1_6 <= 1
 0 <= x_1_7 <= 1
 0 <= x_1_8 <= 1
 0 <= x_1_9 <= 1
 0 <= x_2_0 <= 1
 0 <= x_2_1 <= 1
 0 <= x_2_3 <= 1
 0 <= x_2_4 <= 1
 0 <= x_2_5 <= 1
 0 <= x_2_6 <= 1
 0 <= x_2_7 <= 1
 0 <= x_2_8 <= 1
 0 <= x_2_9 <= 1
 0 <= x_3_0 <= 1
 0 <= x_3_1 <= 1
 0 <= x_3_2 <= 1
 0 <= x_3_4 <= 1
 0 <= x_3_5 <= 1
 0 <= x_3_6 <= 1
 0 <= x_3_7 <= 1
 0 <= x_3_8 <= 1
 0 <= x_3_9 <= 1
 0 <= x_4_0 <= 1
 0 <= x_4_1 <= 1
 0 <= x_4_2 <= 1
 0 <= x_4_3 <= 1
 0 <= x_4_5 <= 1
 0 <= x_4_6 <= 1
 0 <= x_4_7 <= 1
 0 <= x_4_8 <= 1
 0 <= x_4_9 <= 1
 0 <= x_5_0 <= 1
 0 <= x_5_1 <= 1
 0 <= x_5_2 <= 1
 0 <= x_5_3 <= 1
 0 <= x_5_4 <= 1
 0 <= x_5_6 <= 1
 0 <= x_5_7 <= 1
 0 <= x_5_8 <= 1
 0 <= x_5_9 <= 1
 0 <= x_6_0 <= 1
 0 <= x_6_1 <= 1
 0 <= x_6_2 <= 1
 0 <= x_6_3 <= 1
 0 <= x_6_4 <= 1
 0 <= x_6_5 <= 1
 0 <= x_6_7 <= 1
 0 <= x_6_8 <= 1
 0 <= x_6_9 <= 1
 0 <= x_7_0 <= 1
 0 <= x_7_1 <= 1
 0 <= x_7_2 <= 1
 0 <= x_7_3 <= 1
 0 <= x_7_4 <= 1
 0 <= x_7_5 <= 1
 0 <= x_7_6 <= 1
 0 <= x_7_8 <= 1
 0 <= x_7_9 <= 1
 0 <= x_8_0 <= 1
 0 <= x_8_1 <= 1
 0 <= x_8_2 <= 1
 0 <= x_8_3 <= 1
 0 <= x_8_4 <= 1
 0 <= x_8_5 <= 1
 0 <= x_8_6 <= 1
 0 <= x_8_7 <= 1
 0 <= x_8_9 <= 1
 0 <= x_9_0 <= 1
 0 <= x_9_1 <= 1
 0 <= x_9_2 <= 1
 0 <= x_9_3 <= 1
 0 <= x_9_4 <= 1
 0 <= x_9_5 <= 1
 0 <= x_9_6 <= 1
 0 <= x_9_7 <= 1
 0 <= x_9_8 <= 1

Binaries
 x_0_1 x_0_2 x_0_3 x_0_4 x_0_5 x_0_6 x_0_7 x_0_8 x_0_9 x_1_0 x_1_2 x_1_3 x_1_4
 x_1_5 x_1_6 x_1_7 x_1_8 x_1_9 x_2_0 x_2_1 x_2_3 x_2_4 x_2_5 x_2_6 x_2_7 x_2_8
 x_2_9 x_3_0 x_3_1 x_3_2 x_3_4 x_3_5 x_3_6 x_3_7 x_3_8 x_3_9 x_4_0 x_4_1 x_4_2
 x_4_3 x_4_5 x_4_6 x_4_7 x_4_8 x_4_9 x_5_0 x_5_1 x_5_2 x_5_3 x_5_4 x_5_6 x_5_7
 x_5_8 x_5_9 x_6_0 x_6_1 x_6_2 x_6_3 x_6_4 x_6_5 x_6_7 x_6_8 x_6_9 x_7_0 x_7_1
 x_7_2 x_7_3 x_7_4 x_7_5 x_7_6 x_7_8 x_7_9 x_8_0 x_8_1 x_8_2 x_8_3 x_8_4 x_8_5
 x_8_6 x_8_7 x_8_9 x_9_0 x_9_1 x_9_2 x_9_3 x_9_4 x_9_5 x_9_6 x_9_7 x_9_8
End
