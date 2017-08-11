import matplotlib.pyplot as plt

s = """[DEBUG] i=0, loss=28.609, accuracy=0.000 loss=11.049, accuracy=0.130
[DEBUG] i=10, loss=6.586, accuracy=0.350 loss=14.784, accuracy=0.033
[DEBUG] i=20, loss=5.510, accuracy=0.300 loss=9.616, accuracy=0.098
[DEBUG] i=30, loss=2.327, accuracy=0.550 loss=8.296, accuracy=0.109
[DEBUG] i=40, loss=1.043, accuracy=0.650 loss=9.919, accuracy=0.043
[DEBUG] i=50, loss=0.945, accuracy=0.650 loss=9.753, accuracy=0.043
[DEBUG] i=60, loss=6.082, accuracy=0.550 loss=9.287, accuracy=0.011
[DEBUG] i=70, loss=1.676, accuracy=0.700 loss=8.203, accuracy=0.011
[DEBUG] i=80, loss=3.853, accuracy=0.350 loss=7.978, accuracy=0.022
[DEBUG] i=90, loss=0.815, accuracy=0.700 loss=10.759, accuracy=0.000
[DEBUG] i=100, loss=1.476, accuracy=0.450 loss=8.099, accuracy=0.000
[DEBUG] i=110, loss=1.542, accuracy=0.500 loss=7.032, accuracy=0.022
[DEBUG] i=120, loss=0.864, accuracy=0.750 loss=7.523, accuracy=0.000
[DEBUG] i=130, loss=1.220, accuracy=0.650 loss=7.343, accuracy=0.011
[DEBUG] i=140, loss=1.321, accuracy=0.450 loss=7.344, accuracy=0.011
[DEBUG] i=150, loss=1.877, accuracy=0.500 loss=5.584, accuracy=0.033
[DEBUG] i=160, loss=1.501, accuracy=0.500 loss=6.347, accuracy=0.011
[DEBUG] i=170, loss=1.401, accuracy=0.600 loss=6.314, accuracy=0.022
[DEBUG] i=180, loss=1.301, accuracy=0.600 loss=8.552, accuracy=0.000
[DEBUG] i=190, loss=1.162, accuracy=0.550 loss=6.863, accuracy=0.011
[DEBUG] i=200, loss=1.044, accuracy=0.600 loss=5.250, accuracy=0.043
[DEBUG] i=210, loss=0.971, accuracy=0.500 loss=6.634, accuracy=0.011
[DEBUG] i=220, loss=1.243, accuracy=0.550 loss=6.893, accuracy=0.000
[DEBUG] i=230, loss=1.051, accuracy=0.550 loss=7.515, accuracy=0.000
[DEBUG] i=240, loss=1.677, accuracy=0.650 loss=7.700, accuracy=0.000
[DEBUG] i=250, loss=0.751, accuracy=0.750 loss=5.923, accuracy=0.011
[DEBUG] i=260, loss=0.575, accuracy=0.750 loss=7.619, accuracy=0.000
[DEBUG] i=270, loss=1.729, accuracy=0.250 loss=4.914, accuracy=0.043
[DEBUG] i=280, loss=0.780, accuracy=0.700 loss=7.619, accuracy=0.000
[DEBUG] i=290, loss=1.139, accuracy=0.700 loss=6.833, accuracy=0.000
[DEBUG] i=300, loss=0.602, accuracy=0.800 loss=7.246, accuracy=0.000
[DEBUG] i=310, loss=1.052, accuracy=0.550 loss=7.078, accuracy=0.000
[DEBUG] i=320, loss=1.110, accuracy=0.700 loss=6.108, accuracy=0.000
[DEBUG] i=330, loss=0.757, accuracy=0.650 loss=7.097, accuracy=0.000
[DEBUG] i=340, loss=1.075, accuracy=0.500 loss=5.403, accuracy=0.043
[DEBUG] i=350, loss=0.822, accuracy=0.800 loss=7.556, accuracy=0.000
[DEBUG] i=360, loss=0.662, accuracy=0.850 loss=7.143, accuracy=0.000
[DEBUG] i=370, loss=1.211, accuracy=0.600 loss=7.656, accuracy=0.000
[DEBUG] i=380, loss=0.947, accuracy=0.650 loss=6.132, accuracy=0.000
[DEBUG] i=390, loss=1.178, accuracy=0.600 loss=8.606, accuracy=0.000
[DEBUG] i=400, loss=0.721, accuracy=0.800 loss=6.595, accuracy=0.000
[DEBUG] i=410, loss=0.928, accuracy=0.600 loss=6.638, accuracy=0.000
[DEBUG] i=420, loss=0.882, accuracy=0.700 loss=8.227, accuracy=0.000
[DEBUG] i=430, loss=0.950, accuracy=0.600 loss=7.692, accuracy=0.000
[DEBUG] i=440, loss=0.656, accuracy=0.750 loss=6.579, accuracy=0.000
[DEBUG] i=450, loss=0.696, accuracy=0.700 loss=7.569, accuracy=0.000
[DEBUG] i=460, loss=1.237, accuracy=0.500 loss=6.852, accuracy=0.000
[DEBUG] i=470, loss=1.042, accuracy=0.650 loss=5.905, accuracy=0.011
[DEBUG] i=480, loss=1.045, accuracy=0.700 loss=9.507, accuracy=0.000
[DEBUG] i=490, loss=0.835, accuracy=0.700 loss=6.978, accuracy=0.000
[DEBUG] i=500, loss=0.647, accuracy=0.700 loss=5.110, accuracy=0.033
[DEBUG] i=510, loss=0.713, accuracy=0.750 loss=5.993, accuracy=0.000
[DEBUG] i=520, loss=0.600, accuracy=0.700 loss=8.024, accuracy=0.000
[DEBUG] i=530, loss=0.747, accuracy=0.700 loss=6.485, accuracy=0.000
[DEBUG] i=540, loss=0.575, accuracy=0.800 loss=7.991, accuracy=0.000
[DEBUG] i=550, loss=0.496, accuracy=0.850 loss=6.538, accuracy=0.000
[DEBUG] i=560, loss=0.797, accuracy=0.650 loss=6.885, accuracy=0.000
[DEBUG] i=570, loss=0.737, accuracy=0.800 loss=7.661, accuracy=0.000
[DEBUG] i=580, loss=0.701, accuracy=0.750 loss=9.460, accuracy=0.000
[DEBUG] i=590, loss=0.695, accuracy=0.750 loss=6.543, accuracy=0.011
[DEBUG] i=600, loss=1.030, accuracy=0.700 loss=8.976, accuracy=0.000
[DEBUG] i=610, loss=0.840, accuracy=0.700 loss=6.577, accuracy=0.000
[DEBUG] i=620, loss=0.596, accuracy=0.900 loss=6.719, accuracy=0.011
[DEBUG] i=630, loss=1.093, accuracy=0.700 loss=7.559, accuracy=0.000
[DEBUG] i=640, loss=0.543, accuracy=0.800 loss=6.828, accuracy=0.000
[DEBUG] i=650, loss=1.241, accuracy=0.500 loss=5.949, accuracy=0.011
[DEBUG] i=660, loss=1.035, accuracy=0.600 loss=6.366, accuracy=0.000
[DEBUG] i=670, loss=1.511, accuracy=0.350 loss=6.979, accuracy=0.000
[DEBUG] i=680, loss=0.929, accuracy=0.650 loss=7.800, accuracy=0.000
[DEBUG] i=690, loss=0.657, accuracy=0.750 loss=7.476, accuracy=0.000
[DEBUG] i=700, loss=0.687, accuracy=0.600 loss=8.014, accuracy=0.000
[DEBUG] i=710, loss=0.528, accuracy=0.800 loss=9.902, accuracy=0.000
[DEBUG] i=720, loss=1.127, accuracy=0.450 loss=5.689, accuracy=0.033
[DEBUG] i=730, loss=0.523, accuracy=0.900 loss=8.898, accuracy=0.000
[DEBUG] i=740, loss=0.485, accuracy=0.800 loss=6.411, accuracy=0.011
[DEBUG] i=750, loss=0.810, accuracy=0.650 loss=6.835, accuracy=0.000
[DEBUG] i=760, loss=0.964, accuracy=0.650 loss=7.563, accuracy=0.000
[DEBUG] i=770, loss=0.749, accuracy=0.700 loss=7.339, accuracy=0.000
[DEBUG] i=780, loss=1.196, accuracy=0.600 loss=8.418, accuracy=0.000
[DEBUG] i=790, loss=0.640, accuracy=0.800 loss=7.573, accuracy=0.000
[DEBUG] i=800, loss=0.618, accuracy=0.700 loss=7.630, accuracy=0.011
[DEBUG] i=810, loss=0.790, accuracy=0.700 loss=9.899, accuracy=0.000
[DEBUG] i=820, loss=0.630, accuracy=0.650 loss=8.292, accuracy=0.000
[DEBUG] i=830, loss=0.743, accuracy=0.700 loss=9.370, accuracy=0.000
[DEBUG] i=840, loss=0.806, accuracy=0.700 loss=6.484, accuracy=0.011
[DEBUG] i=850, loss=0.850, accuracy=0.650 loss=8.341, accuracy=0.000
[DEBUG] i=860, loss=0.534, accuracy=0.850 loss=5.385, accuracy=0.011
[DEBUG] i=870, loss=0.785, accuracy=0.700 loss=5.699, accuracy=0.011
[DEBUG] i=880, loss=0.589, accuracy=0.650 loss=6.338, accuracy=0.000
[DEBUG] i=890, loss=0.412, accuracy=0.900 loss=6.679, accuracy=0.000
[DEBUG] i=900, loss=1.154, accuracy=0.600 loss=7.733, accuracy=0.000
[DEBUG] i=910, loss=0.510, accuracy=0.800 loss=8.551, accuracy=0.000
[DEBUG] i=920, loss=0.708, accuracy=0.750 loss=7.145, accuracy=0.000
[DEBUG] i=930, loss=0.435, accuracy=0.850 loss=6.951, accuracy=0.000
[DEBUG] i=940, loss=0.543, accuracy=0.800 loss=6.174, accuracy=0.011
[DEBUG] i=950, loss=0.830, accuracy=0.650 loss=7.033, accuracy=0.000
[DEBUG] i=960, loss=0.978, accuracy=0.600 loss=6.797, accuracy=0.000
[DEBUG] i=970, loss=0.986, accuracy=0.650 loss=6.802, accuracy=0.011
[DEBUG] i=980, loss=0.470, accuracy=0.750 loss=7.772, accuracy=0.000
[DEBUG] i=990, loss=0.771, accuracy=0.550 loss=7.177, accuracy=0.000"""
s = s.split("\n")
assert len(s) == 100, len(s)
s.sort(key=lambda x: float(x.split(" ")[4].split("=")[1].rstrip(",")))
for x in s:
    print(x)

x = list(map(lambda _x: int(_x.split(" ")[1].split("=")[1].rstrip(",")), s))
y = list(map(lambda _x: float(_x.split(" ")[5].split("=")[1]), s))
plt.scatter(x, y)
plt.show()
