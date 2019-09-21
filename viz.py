import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt


p_c = "writeup_images/center_2016_12_01_13_36_47_622.jpg"
p_l = "writeup_images/left_2016_12_01_13_36_47_622.jpg"
p_r = "writeup_images/right_2016_12_01_13_36_47_622.jpg"

i_c = ndimage.imread(p_c)
i_l = ndimage.imread(p_l)
i_r = ndimage.imread(p_r)

ai_c = np.fliplr(i_c)
ai_l = np.fliplr(i_l)
ai_r = np.fliplr(i_r)

images = [i_c, i_l, i_r, ai_c, ai_l, ai_r]


# Plot the result
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 3))
f.tight_layout()

ax1.imshow(i_c)
ax1.set_title('Center, Steering=-0.05975719', fontsize=10)

ax2.imshow(i_l)
ax2.set_title('Left, Steering=1.1855326', fontsize=10)

ax3.imshow(i_r)
ax3.set_title('Right, Steering=-0.2', fontsize=10)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()