# Hawkeye juggling video analysis
Hawkeye is a Python/Qt application to analyze and view videos containing juggling.

The application uses the OpenCV computer vision library to identify and track balls moving in parabolic trajectories, displaying the information as an overlay on top of a video viewer. Hawkeye's video viewer supports smooth stepping forward/backward by single frames, as well as zooming, to allow you to see details well.

The goal of Hawkeye is to help jugglers understand their form and improve their skills by allowing them to view practice video easily and efficiently.

### Application downloads
[Hawkeye-1.0.dmg](https://storage.googleapis.com/hawkeye-dl/Hawkeye-1.0.dmg) -- Mac OS X (Mac OS 10.11 or above) _(Note: If launching the app gives an "identity of the developer cannot be confirmed" message, right-click on the app and select Open, then select Open in the dialog box. After it launches successfully the warning message won't appear again.)_

[Hawkeye-1.0.exe](https://storage.googleapis.com/hawkeye-dl/Hawkeye-1.0.exe) -- Windows installer

### Sample videos
Below are some sample videos to demo Hawkeye in action. Drag and drop a video onto the Videos: box in the application, and note that it may take a few minutes to analyze videos of this length.

- [JB 5 balls](https://storage.googleapis.com/hawkeye-dl/juggling_test_5.mov)
- [AG 9 balls](https://storage.googleapis.com/hawkeye-dl/TBTB3_9balls.mov)
- [AG 8 balls](https://storage.googleapis.com/hawkeye-dl/TBTB3_8balls.mov)

### Tracking details
The key challenge in this project was to build an object tracker that doesn't rely on chroma keying, object tagging, or any special recording equipment. The goal was to be able to track thrown objects in ordinary video footage, captured with ordinary cameras, under a variety of recording conditions.

The general problem of tracking objects in unconstrained video footage is quite difficult, and state of the art algorithms often give many false positives and false negatives, identifying objects that don't exist and missing (or losing) ones that do. What makes this problem more tractable is the degree of predictability in a thrown object's path: If we have many observations that together fit well to a parabolic trajectory, then we can have a high degree of confidence those observations represent a real thrown object. Conversely if an observation doesn't fit into any high-confidence trajectory, we can likely discard it as noise.

At a high level our approach is:
- Use a feature detector to identify moving objects in each frame. For this we use OpenCV's MOG background subtraction algorithm and simple blob detector. The majority of events detected at this stage are noise: Moving objects in the background, the juggler's body movement, or just camera noise.
- Piece together nearby (in space and time) events into preliminary parabolic tracks.
- Optimize those parabolic tracks using the Expectation Maximization (EM) algorithm. This alternates between calculating weights for each event's affiliation with each arc (E step), and weighted least-squares fitting to refine the parabolas (M step). We merge and prune out bad arcs as we go. This is loosely based on Ribnick et al's algorithm (reference below) but we get higher reliability than their published numbers by doing more preprocessing before applying the EM algorithm.

### References
- Moon, T.K., "The Expectation Maximization Algorithm”, IEEE Signal Processing Magazine, vol. 13, no. 6, pp. 47–60, November 1996.
- Ribnick, E. et al, "Detection of Thrown Objects in Indoor and Outdoor Scenes", Proceedings of the 2007 IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2007.
