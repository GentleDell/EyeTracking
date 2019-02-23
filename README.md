# EyeTracking
Tracking eyes to enable eyes-screen interaction


We have adjusted the structure of [eyelike](https://github.com/trishume/eyeLike) and embedded the optimization method to it as a Class. However, since there are several hidden assumptions behind this optimization method, we need iterate the method many times. Main tasks are listed below:
1. Loosen the assumption that users' heads move slowly when they use their computers. ---- choose a higher threshold for the median filter (with reliable reference)

2. Recognize the user and regardless of other faces. ---- distinct user's face from background faces using the screen-face-distance.

3. Allow the face detection part to have a larger dynamic detection range. ---- adjust an input value of the detection function provided by OpenCV. 

4. Map eye poses to the corresponding parts of the screen where users are looking at.
