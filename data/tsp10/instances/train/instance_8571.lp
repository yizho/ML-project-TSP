\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: TSP

Minimize
 obj: 68.666731844226 x_0_1 + 35.393504874466 x_0_2 + 93.476815996944 x_0_3
      + 51.428009400853 x_0_4 + 31.947348261537 x_0_5 + 29.778858361051 x_0_6
      + 125.369316128305 x_0_7 + 75.826207232189 x_0_8 + 60.692525025636 x_0_9
      + 68.666731844226 x_1_0 + 48.363205563547 x_1_2 + 59.289252230585 x_1_3
      + 17.239189245191 x_1_4 + 43.888514907413 x_1_5 + 39.384200136384 x_1_6
      + 84.506336336333 x_1_7 + 45.071731500010 x_1_8 + 20.663883616160 x_1_9
      + 35.393504874466 x_2_0 + 48.363205563547 x_2_1 + 58.134608037519 x_2_3
      + 34.271963687126 x_2_4 + 7.811587503925 x_2_5 + 19.694223476975 x_2_6
      + 90.013401565286 x_2_7 + 40.769591529462 x_2_8 + 32.154412080419 x_2_9
      + 93.476815996944 x_3_0 + 59.289252230585 x_3_1 + 58.134608037519 x_3_2
      + 62.664503004099 x_3_4 + 62.196288543604 x_3_5 + 69.794451516727 x_3_6
      + 31.894625649901 x_3_7 + 18.325863623225 x_3_8 + 43.850963080759 x_3_9
      + 51.428009400853 x_4_0 + 17.239189245191 x_4_1 + 34.271963687126 x_4_2
      + 62.664503004099 x_4_3 + 28.439988569826 x_4_5 + 22.291010397971 x_4_6
      + 91.598717179980 x_4_7 + 45.481793161508 x_4_8 + 18.818274322513 x_4_9
      + 31.947348261537 x_5_0 + 43.888514907413 x_5_1 + 7.811587503925 x_5_2
      + 62.196288543604 x_5_3 + 28.439988569826 x_5_4 + 11.883667218246 x_5_6
      + 94.036557346109 x_5_7 + 44.198477317604 x_5_8 + 30.365047280185 x_5_9
      + 29.778858361051 x_6_0 + 39.384200136384 x_6_1 + 19.694223476975 x_6_2
      + 69.794451516727 x_6_3 + 22.291010397971 x_6_4 + 11.883667218246 x_6_5
      + 101.204220234425 x_6_7 + 51.473851910164 x_6_8 + 31.637432923259 x_6_9
      + 125.369316128305 x_7_0 + 84.506336336333 x_7_1 + 90.013401565286 x_7_2
      + 31.894625649901 x_7_3 + 91.598717179980 x_7_4 + 94.036557346109 x_7_5
      + 101.204220234425 x_7_6 + 49.909873103192 x_7_8 + 73.077054696083 x_7_9
      + 75.826207232189 x_8_0 + 45.071731500010 x_8_1 + 40.769591529462 x_8_2
      + 18.325863623225 x_8_3 + 45.481793161508 x_8_4 + 44.198477317604 x_8_5
      + 51.473851910164 x_8_6 + 49.909873103192 x_8_7 + 26.967729296261 x_8_9
      + 60.692525025636 x_9_0 + 20.663883616160 x_9_1 + 32.154412080419 x_9_2
      + 43.850963080759 x_9_3 + 18.818274322513 x_9_4 + 30.365047280185 x_9_5
      + 31.637432923259 x_9_6 + 73.077054696083 x_9_7 + 26.967729296261 x_9_8
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
