\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: TSP

Minimize
 obj: 79.618105973756 x_0_1 + 39.783027234049 x_0_2 + 65.004168664372 x_0_3
      + 58.283816260848 x_0_4 + 87.484033325310 x_0_5 + 30.286418426914 x_0_6
      + 63.054562991778 x_0_7 + 29.349224633588 x_0_8 + 53.711233539446 x_0_9
      + 32.109653100628 x_0_10 + 28.051708225294 x_0_11 + 64.753891398570 x_0_12
      + 78.768976814433 x_0_13 + 96.497676847140 x_0_14 + 79.618105973756 x_1_0
      + 103.995219279547 x_1_2 + 23.348085836160 x_1_3 + 75.216642635854 x_1_4
      + 53.361609529251 x_1_5 + 100.464287856266 x_1_6 + 42.605045565489 x_1_7
      + 76.196948518092 x_1_8 + 37.408865868909 x_1_9 + 50.497012534213 x_1_10
      + 80.271171255142 x_1_11 + 22.631342150326 x_1_12 + 34.791175304720 x_1_13
      + 72.034686544896 x_1_14 + 39.783027234049 x_2_0 + 103.995219279547 x_2_1
      + 83.303892175625 x_2_3 + 46.953801528595 x_2_4 + 89.802338362043 x_2_5
      + 10.904331165029 x_2_6 + 71.571114193369 x_2_7 + 27.919567047261 x_2_8
      + 68.688271414934 x_2_9 + 66.872456348598 x_2_10 + 23.729114156353 x_2_11
      + 83.550949150907 x_2_12 + 89.356821596819 x_2_13 + 90.990409619663 x_2_14
      + 65.004168664372 x_3_0 + 23.348085836160 x_3_1 + 83.303892175625 x_3_2
      + 51.868848911386 x_3_4 + 36.849372604673 x_3_5 + 81.149272458139 x_3_6
      + 19.709558273273 x_3_7 + 55.412288518313 x_3_8 + 14.792220451428 x_3_9
      + 42.389747142309 x_3_10 + 59.734175760412 x_3_11 + 1.039644001004 x_3_12
      + 19.283484912382 x_3_13 + 54.427419108264 x_3_14 + 58.283816260848 x_4_0
      + 75.216642635854 x_4_1 + 46.953801528595 x_4_2 + 51.868848911386 x_4_3
      + 44.377882151604 x_4_5 + 51.596826445645 x_4_6 + 33.524971708367 x_4_7
      + 29.608358089281 x_4_8 + 38.650243117035 x_4_9 + 63.237807881566 x_4_10
      + 32.581834561960 x_4_11 + 52.598545470727 x_4_12 + 49.211217405463 x_4_13
      + 44.076301379446 x_4_14 + 87.484033325310 x_5_0 + 53.361609529251 x_5_1
      + 89.802338362043 x_5_2 + 36.849372604673 x_5_3 + 44.377882151604 x_5_4
      + 92.159656317806 x_5_6 + 25.178376989421 x_5_7 + 65.943158771093 x_5_8
      + 37.193350041541 x_5_9 + 74.643427878331 x_5_10 + 70.216335042584 x_5_11
      + 37.869667719621 x_5_12 + 18.772731270155 x_5_13 + 18.690518225680 x_5_14
      + 30.286418426914 x_6_0 + 100.464287856266 x_6_1 + 10.904331165029 x_6_2
      + 81.149272458139 x_6_3 + 51.596826445645 x_6_4 + 92.159656317806 x_6_5
      + 71.768729395200 x_6_7 + 26.803961905748 x_6_8 + 67.011327711367 x_6_9
      + 59.594754311074 x_6_10 + 22.335484682418 x_6_11 + 81.263545299391 x_6_12
      + 89.467117804359 x_6_13 + 95.434598182389 x_6_14 + 63.054562991778 x_7_0
      + 42.605045565489 x_7_1 + 71.571114193369 x_7_2 + 19.709558273273 x_7_3
      + 33.524971708367 x_7_4 + 25.178376989421 x_7_5 + 71.768729395200 x_7_6
      + 44.965287014324 x_7_8 + 12.611770226814 x_7_9 + 49.944524523664 x_7_10
      + 49.438699550767 x_7_11 + 20.626725048033 x_7_12 + 17.794887920451 x_7_13
      + 39.103843278767 x_7_14 + 29.349224633588 x_8_0 + 76.196948518092 x_8_1
      + 27.919567047261 x_8_2 + 55.412288518313 x_8_3 + 29.608358089281 x_8_4
      + 65.943158771093 x_8_5 + 26.803961905748 x_8_6 + 44.965287014324 x_8_7
      + 40.872944240367 x_8_9 + 44.298604265868 x_8_10 + 4.480515819069 x_8_11
      + 55.643857575319 x_8_12 + 62.678181137972 x_8_13 + 71.153500518667 x_8_14
      + 53.711233539446 x_9_0 + 37.408865868909 x_9_1 + 68.688271414934 x_9_2
      + 14.792220451428 x_9_3 + 38.650243117035 x_9_4 + 37.193350041541 x_9_5
      + 67.011327711367 x_9_6 + 12.611770226814 x_9_7 + 40.872944240367 x_9_8
      + 37.493055321929 x_9_10 + 45.255928726029 x_9_11 + 15.203081690018 x_9_12
      + 25.119871475091 x_9_13 + 51.713236696424 x_9_14 + 32.109653100628 x_10_0
      + 50.497012534213 x_10_1 + 66.872456348598 x_10_2 + 42.389747142309 x_10_3
      + 63.237807881566 x_10_4 + 74.643427878331 x_10_5 + 59.594754311074 x_10_6
      + 49.944524523664 x_10_7 + 44.298604265868 x_10_8 + 37.493055321929 x_10_9
      + 46.393530976075 x_10_11 + 41.736795654297 x_10_12
      + 60.413847530865 x_10_13 + 88.766279030903 x_10_14
      + 28.051708225294 x_11_0 + 80.271171255142 x_11_1 + 23.729114156353 x_11_2
      + 59.734175760412 x_11_3 + 32.581834561960 x_11_4 + 70.216335042584 x_11_5
      + 22.335484682418 x_11_6 + 49.438699550767 x_11_7 + 4.480515819069 x_11_8
      + 45.255928726029 x_11_9 + 46.393530976075 x_11_10
      + 59.944248581534 x_11_12 + 67.157182695750 x_11_13
      + 74.945568924134 x_11_14 + 64.753891398570 x_12_0
      + 22.631342150326 x_12_1 + 83.550949150907 x_12_2 + 1.039644001004 x_12_3
      + 52.598545470727 x_12_4 + 37.869667719621 x_12_5 + 81.263545299391 x_12_6
      + 20.626725048033 x_12_7 + 55.643857575319 x_12_8 + 15.203081690018 x_12_9
      + 41.736795654297 x_12_10 + 59.944248581534 x_12_11
      + 20.226832873899 x_12_13 + 55.465577912257 x_12_14
      + 78.768976814433 x_13_0 + 34.791175304720 x_13_1 + 89.356821596819 x_13_2
      + 19.283484912382 x_13_3 + 49.211217405463 x_13_4 + 18.772731270155 x_13_5
      + 89.467117804359 x_13_6 + 17.794887920451 x_13_7 + 62.678181137972 x_13_8
      + 25.119871475091 x_13_9 + 60.413847530865 x_13_10
      + 67.157182695750 x_13_11 + 20.226832873899 x_13_12
      + 37.330151197114 x_13_14 + 96.497676847140 x_14_0
      + 72.034686544896 x_14_1 + 90.990409619663 x_14_2 + 54.427419108264 x_14_3
      + 44.076301379446 x_14_4 + 18.690518225680 x_14_5 + 95.434598182389 x_14_6
      + 39.103843278767 x_14_7 + 71.153500518667 x_14_8 + 51.713236696424 x_14_9
      + 88.766279030903 x_14_10 + 74.945568924134 x_14_11
      + 55.465577912257 x_14_12 + 37.330151197114 x_14_13
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
